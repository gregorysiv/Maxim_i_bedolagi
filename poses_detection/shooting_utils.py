import numpy as np
from collections import deque


class ShootingPoseAnalyzer:
    """
    Класс для комплексного анализа позы стрелка с учетом истории и трендов.
    """

    def __init__(self, window_size=10, confirmation_frames=5):
        """
        Args:
            window_size (int): Размер окна для анализа истории.
            confirmation_frames (int): Количество кадров для подтверждения позы.
        """
        self.window_size = window_size
        self.confirmation_frames = confirmation_frames
        self.history = deque(maxlen=window_size)
        self.angle_history = deque(maxlen=window_size)
        self.stance_history = deque(maxlen=window_size)

    def calculate_stability_score(self, points, confs, confidence_threshold=0.5):
        """
        Вычисляет оценку устойчивости стойки.

        Returns:
            float: Оценка устойчивости от 0 до 1.
        """
        L_ANKLE, R_ANKLE = 15, 16
        L_KNEE, R_KNEE = 13, 14
        L_HIP, R_HIP = 11, 12

        score = 0.0
        factors = 0

        # Фактор 1: Ширина стойки
        if confs[L_ANKLE] > confidence_threshold and confs[R_ANKLE] > confidence_threshold:
            stance_width = abs(points[L_ANKLE][0] - points[R_ANKLE][0])
            # Нормализуем: 50px = 0.5, 100px = 1.0
            width_score = min(stance_width / 100, 1.0)
            score += width_score
            factors += 1

        # Фактор 2: Симметрия высоты ног
        if (
            confs[L_ANKLE] > confidence_threshold
            and confs[R_ANKLE] > confidence_threshold
            and confs[L_HIP] > confidence_threshold
            and confs[R_HIP] > confidence_threshold
        ):
            left_leg_height = abs(points[L_HIP][1] - points[L_ANKLE][1])
            right_leg_height = abs(points[R_HIP][1] - points[R_ANKLE][1])

            if left_leg_height > 0 and right_leg_height > 0:
                height_ratio = min(left_leg_height, right_leg_height) / max(
                    left_leg_height, right_leg_height
                )
                score += height_ratio
                factors += 1

        # Фактор 3: Углы в коленях (слегка согнуты для устойчивости)
        if (
            confs[L_KNEE] > confidence_threshold
            and confs[L_HIP] > confidence_threshold
            and confs[L_ANKLE] > confidence_threshold
        ):
            # Используем теорему косинусов
            a = np.linalg.norm(np.array(points[L_HIP]) - np.array(points[L_KNEE]))
            b = np.linalg.norm(np.array(points[L_KNEE]) - np.array(points[L_ANKLE]))
            c = np.linalg.norm(np.array(points[L_HIP]) - np.array(points[L_ANKLE]))

            if a > 0 and b > 0:
                cos_angle = (a**2 + b**2 - c**2) / (2 * a * b)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                knee_angle = np.degrees(np.arccos(cos_angle))

                # Идеальный угол для устойчивости: 140-170°
                if 140 <= knee_angle <= 170:
                    knee_score = 1.0
                elif 120 <= knee_angle < 140 or knee_angle > 170:
                    knee_score = 0.7
                else:
                    knee_score = 0.3

                score += knee_score
                factors += 1

        return score / factors if factors > 0 else 0.0

    def analyze_weapon_alignment(self, points, confs, shooting_arm="right", confidence_threshold=0.5):
        """
        Анализирует выравнивание 'оружия' (линии глаз-запястье).

        Returns:
            dict: Метрики выравнивания.
        """
        NOSE = 0
        L_SHOULDER, R_SHOULDER = 5, 6
        L_WRIST, R_WRIST = 9, 10

        alignment = {
            "horizontal_alignment": 0.0,
            "vertical_alignment": 0.0,
            "overall_score": 0.0,
        }

        if shooting_arm == "right":
            if confs[NOSE] > confidence_threshold and confs[R_WRIST] > confidence_threshold:
                # Горизонтальное выравнивание (предплечье должно быть горизонтально)
                if confs[R_SHOULDER] > confidence_threshold:
                    forearm_tilt = abs(points[R_WRIST][1] - points[R_SHOULDER][1])
                    alignment["horizontal_alignment"] = max(0, 1 - forearm_tilt / 50)

                # Вертикальное выравнивание (взгляд вдоль руки)
                eye_to_wrist_vector = np.array(points[R_WRIST]) - np.array(points[NOSE])
                if np.linalg.norm(eye_to_wrist_vector) > 0:
                    # Идеальный угол между горизонталью и направлением взгляда ~0°
                    horizontal_vector = np.array([100, 0])
                    dot_product = np.dot(eye_to_wrist_vector[:2], horizontal_vector)
                    norm_product = np.linalg.norm(eye_to_wrist_vector[:2]) * np.linalg.norm(horizontal_vector)

                    if norm_product > 0:
                        cos_angle = dot_product / norm_product
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)
                        alignment["vertical_alignment"] = (cos_angle + 1) / 2
        else:
            if confs[NOSE] > confidence_threshold and confs[L_WRIST] > confidence_threshold:
                if confs[L_SHOULDER] > confidence_threshold:
                    forearm_tilt = abs(points[L_WRIST][1] - points[L_SHOULDER][1])
                    alignment["horizontal_alignment"] = max(0, 1 - forearm_tilt / 50)

                eye_to_wrist_vector = np.array(points[L_WRIST]) - np.array(points[NOSE])
                if np.linalg.norm(eye_to_wrist_vector) > 0:
                    horizontal_vector = np.array([100, 0])
                    dot_product = np.dot(eye_to_wrist_vector[:2], horizontal_vector)
                    norm_product = np.linalg.norm(eye_to_wrist_vector[:2]) * np.linalg.norm(horizontal_vector)

                    if norm_product > 0:
                        cos_angle = dot_product / norm_product
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)
                        alignment["vertical_alignment"] = (cos_angle + 1) / 2

        alignment["overall_score"] = (alignment["horizontal_alignment"] + alignment["vertical_alignment"]) / 2

        return alignment

    def is_consistent_shooting_stance(self, points, confs, confidence_threshold=0.5):
        """
        Проверяет, является ли поза устойчивой стойкой стрелка.

        Returns:
            tuple: (is_stance, details)
        """
        from poses_detection.shooter_pose import check_shooter_pose

        # Проверяем текущую позу
        result = check_shooter_pose(points, confs, confidence_threshold)
        current_is_shooter = result != ""

        # Добавляем в историю
        self.history.append(current_is_shooter)

        # Анализируем историю
        if len(self.history) >= self.confirmation_frames:
            recent = list(self.history)[-self.confirmation_frames:]
            detection_rate = sum(recent) / len(recent)

            # Требуем высокую согласованность
            if detection_rate >= 0.8:
                details = {
                    "detection_rate": detection_rate,
                    "frames_analyzed": len(self.history),
                    "consistent": True,
                }
                return True, details

        details = {
            "detection_rate": sum(self.history) / len(self.history) if self.history else 0,
            "frames_analyzed": len(self.history),
            "consistent": False,
        }

        return False, details
