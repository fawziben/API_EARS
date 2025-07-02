from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import math
from ultralytics import YOLO
import tensorflow as tf
# from tensorflow.keras.models import load_model
from PIL import Image
import mediapipe as mp

# Initialiser Flask et autoriser les CORS
app = Flask(__name__)
CORS(app)

# ============================ Chargement des modèles ============================
yolo_model_path = 'models\model_yolo8_small.pt'
# ajouter votre modèle ici
landmark_model_path = 'models\model_landmark_lite.tflite'

yolo_model = YOLO(yolo_model_path)

interpreter = tf.lite.Interpreter(model_path=landmark_model_path)
interpreter.allocate_tensors()
# Récupérer les détails des tenseurs d'entrée et de sortie
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ============================ Détection faciale ============================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True)


def calculate_head_orientation(image):
    """
    Calcule les angles d'orientation (Yaw, Pitch, Roll) de la tête basée sur les landmarks du visage.
    """
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # Points clés pour les calculs
            nose = landmarks[1]
            left_eye_outer = landmarks[33]
            right_eye_outer = landmarks[263]
            chin = landmarks[152]

            # Yaw Angle (orientation horizontale gauche-droite)
            eye_center_x = (left_eye_outer.x + right_eye_outer.x) / 2
            eye_center_y = (left_eye_outer.y + right_eye_outer.y) / 2
            yaw_angle = math.degrees(math.atan2(
                nose.x - eye_center_x, nose.y - eye_center_y))

            # Pitch Angle (orientation haut-bas)
            pitch_angle = math.degrees(math.atan2(
                nose.y - chin.y, nose.z - chin.z))

            # Roll Angle (inclinaison de la tête)
            roll_angle = math.degrees(math.atan2(left_eye_outer.y - right_eye_outer.y,
                                                 left_eye_outer.x - right_eye_outer.x))

            return {
                "yaw": round(yaw_angle, 2),
                "pitch": round(pitch_angle, 2),
                "roll": round(roll_angle, 2)
            }
    return None

# ============================ Préparer le modèle landmarks ======================================


def predict_landmarks_tflite(img_np):
    # img_np est un tableau NumPy de forme (1, 224, 224, 1)
    interpreter.set_tensor(
        input_details[0]['index'], img_np.astype(np.float32))
    interpreter.invoke()        # Lance l'inférence

    output_data = interpreter.get_tensor(
        output_details[0]['index'])        # Récupère la sortie
    # output_data est maintenant un tableau NumPy contenant la prédiction
    return output_data[0]       # shape = (24,)


def prepare_landmark_input(img, bbox):
    """
    Prépare une image (recadrée) pour la prédiction des landmarks.
    """
    x_min, y_min, x_max, y_max = map(int, bbox)
    cropped_img = img[y_min:y_max, x_min:x_max]  # Recadrer l'image
    # Convertir en noir et blanc
    gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray_img, (224, 224))  # Redimensionner à 224x224
    normalized_img = resized_img / 255.0  # Normaliser les pixels entre 0 et 1
    # Ajouter les dimensions nécessaires
    input_img = np.expand_dims(normalized_img, axis=(0, -1))

    return input_img


@app.route('/')
def home():
    return jsonify({"message": "Bienvenue sur l'API de détection des oreilles."}), 200


@app.route('/api/detect-ear', methods=['POST'])
def detect_ear():
    try:
        # Vérifier si une image est envoyée
        if 'image' not in request.files:
            return jsonify({"error": "Aucune image n'a été envoyée"}), 400

        # Lire l'image envoyée
        image_file = request.files['image']
        image = Image.open(image_file)
        image_np = np.array(image)

        # Vérifier si l'image est en RGB (sinon la convertir) avec 2 ou 3 canals (v1)
        # On doit aussi traiter le cas ou l'on a 1 ou 4 canaux (type RGBA, ex: photo .png)
        if len(image_np.shape) == 2:
            # Image en niveaux de gris → convertit en RGB
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

        elif image_np.shape[2] == 4:
            # Image RGBA (4 canaux) → convertit en RGB (3 canaux)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

        # Détection des oreilles avec YOLOv8
        # seuil de confiance conf=0.1
        results = yolo_model.predict(source=image_np, conf=0.1)
        predictions = []
        # Calcul de l'orientation de la tête entière
        orientation = calculate_head_orientation(image_np)
        print("Détections YOLO :", len(results[0].boxes))

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            bboxes = results[0].boxes.xyxy.cpu().numpy()

            # Parcourir chaque bounding box détectée
            for bbox in bboxes:
                # Préparer l'image pour les landmarks
                input_img = prepare_landmark_input(image_np, bbox)

                # Prédire les landmarks
                landmarks = predict_landmarks_tflite(input_img)
                landmarks_x = landmarks[:12]  # x-coordonnées
                landmarks_y = landmarks[12:]  # y-coordonnées       +++++++++++

                # Convertir les landmarks en coordonnées globales
                global_landmarks = [
                    [int(bbox[0] + x * (bbox[2] - bbox[0])),
                     int(bbox[1] + y * (bbox[3] - bbox[1]))]
                    for x, y in zip(landmarks_x, landmarks_y)
                ]

                # Indices des landmarks pour le lobe et le sommet de l'oreille
                lobe_index = 0  # Lobe 1
                top_outer_helix_index = 4  # Top Outer Helix

                lobe = global_landmarks[lobe_index]
                top_outer_helix = global_landmarks[top_outer_helix_index]

                # Calcul de la distance entre le lobe et le sommet
                distance_px = math.sqrt(
                    (top_outer_helix[0] - lobe[0])**2 + (top_outer_helix[1] - lobe[1])**2)

                landmark_names = [
                    "Lobe 1", "Tragus", "Industrial",
                    "Helix 1", "Top Outer Helix", "Helix 2",
                    "Helix 3", "Helix 4", "Helix 5",
                    "Helix 6", "Lobe 2", "Lobe 3"
                ]

                # Indices des landmarks pour tragus et anti-tragus (selon votre mapping)
                tragus_index = 1  # Tragus
                lobe_3_index = 11  # Troisième trou au niveau du lobe
                tragus = global_landmarks[tragus_index]
                # on récupère les coord
                lobe_3 = global_landmarks[lobe_3_index]

                # Déterminer si l'oreille est gauche ou droite
                if tragus[0] < lobe_3[0]:  # Si tragus est à gauche du troisième trou de loreille
                    ear_side = "left"
                else:
                    ear_side = "right"

                # Ajouter les prédictions au résultat
                predictions.append({
                    "ear_side": ear_side,
                    "landmarks": [
                        {"name": name, "coordinates": coord}
                        for name, coord in zip(landmark_names, global_landmarks)
                    ],
                    "distance_px": distance_px,  # distance entre le lobe et le sommet de l'oreille
                    "confidence": float(results[0].boxes.conf[0])
                })

        # Si aucune bounding box n'est détectée
        if not predictions:
            return jsonify({"status": "no_detection", "predictions": []})

        print(predictions)
        # Retourner les résultats
        return jsonify({
            "status": "success",
            "predictions": predictions,
            "orientation": orientation
        })

    except Exception as e:
        print(e)
        return jsonify({"error": "Erreur lors du traitement de l'image"}), 500


if __name__ == '__main__':
    app.run(debug=True)
