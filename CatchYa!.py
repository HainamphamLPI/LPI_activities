import warnings
import time
warnings.filterwarnings("ignore", category=UserWarning, module='greenlet')

import cv2
import numpy as np

# Initialize the camera
cap = cv2.VideoCapture(1)

# Screen dimensions
WIDTH, HEIGHT = 500, 700
cursor_size = 30 
score = 0
game_running = False  # Flag to check if the game is running
paused = False
game_over = False

# Initial cursor position
cursor_x = WIDTH // 2
cursor_y = HEIGHT // 2

# List to store trackers and their bounding boxes
trackers = []
trackers_boxes = []

# Timer-related variables
start_time = 0
elapsed_time = 0
last_score_time = 0
countdown_time = 120  # Set to 2 seconds for each score countdown

# Load and blur background image
background_img_path = '/Users/hainam/Desktop/78qpIkbEsTj45.jpg!w700wp.webp' 
background_img = cv2.imread(background_img_path)

if background_img is None:
    print(f"Error loading image from {background_img_path}. Please check the file path.")
    exit()

background_img = cv2.resize(background_img, (WIDTH, HEIGHT))
blurred_background_img = cv2.GaussianBlur(background_img, (41, 41), 0)

# Function to initialize trackers for detected small gray objects
def initialize_trackers(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the color range for green
    lower_color = np.array([35, 50, 50])    # Adjusted lower bound for green
    upper_color = np.array([85, 255, 255])  # Adjusted upper bound for green

    mask = cv2.inRange(hsv, lower_color, upper_color)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 100  # Further reduced threshold for detecting smaller objects

    valid_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]
    valid_contours = valid_contours[:10]

    global trackers, trackers_boxes
    trackers = []
    trackers_boxes = []

    for contour in valid_contours:
        x, y, w, h = cv2.boundingRect(contour)
        bbox = (x, y, w, h)
        tracker = cv2.TrackerCSRT_create()  # Initialize CSRT tracker
        trackers.append(tracker)
        trackers_boxes.append(bbox)

    # Initialize trackers with the detected bounding boxes
    for i, tracker in enumerate(trackers):
        tracker.init(frame, tuple(trackers_boxes[i]))

# Function to display text on the screen
def draw_text(frame, text, position, font_scale=1, thickness=4, color=(255, 255, 255), outline_color=(0, 0, 0), outline_thickness=2):
    # Draw text outline
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline_color, outline_thickness, cv2.LINE_AA)
    # Draw the main text
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

# Function for the game start screen
def start_screen():
    frame = blurred_background_img.copy()
    draw_text(frame, "CatchYa!", (70, 200), font_scale=2, thickness=4, color=(0, 255, 0), outline_color=(0, 0, 0), outline_thickness=6)
    draw_text(frame, "Press 'S' to Start", (60, 300), font_scale=1, color=(255, 255, 255),outline_color=(0, 0, 0), outline_thickness=2)
    draw_text(frame, "Press 'Q' to Quit", (60, 350), font_scale=1, color=(255, 255, 255), outline_color=(0, 0, 0), outline_thickness=2)
    cv2.imshow("CatchYa!", frame)

# Function for the game over screen
def game_over_screen(final_score):
    frame = blurred_background_img.copy()
    draw_text(frame, "Game Over", (100, 200), font_scale=2, thickness=4, color=(0, 0, 255), outline_color=(0, 0, 0), outline_thickness=6)
    draw_text(frame, f"Your Score: {final_score}", (80, 300), font_scale=1, color=(255, 255, 255), outline_color=(0, 0, 0), outline_thickness=2)
    draw_text(frame, "Press 'R' to Restart", (80, 350), font_scale=1, color=(255, 255, 255), outline_color=(0, 0, 0), outline_thickness=2)
    draw_text(frame, "Press 'Q' to Quit", (80, 400), font_scale=1, color=(255, 255, 255), outline_color=(0, 0, 0), outline_thickness=2)
    cv2.imshow("CatchYa!", frame)

# Define 4 additional squares with their positions and sizes
squares = [
    {'x': 100, 'y': 100, 'size': 100},  # Top-left corner
    {'x': 1700, 'y': 100, 'size': 100},  # Top-right corner
    {'x': 100, 'y': 850, 'size': 100},  # Bottom-left corner
    {'x': 1700, 'y': 850, 'size': 100}  # Bottom-right corner
]

# Main game loop
while True:
    if not game_running and not game_over:
        start_screen()  # Display start screen
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            game_running = True
            score = 0
            cursor_x = WIDTH // 2
            cursor_y = HEIGHT // 2
            start_time = time.time()
            last_score_time = start_time
        elif key == ord('q'):
            break

    elif game_running and not paused:
        ret, frame = cap.read()
        if not ret:
            break

        if not trackers:
            initialize_trackers(frame)

        # Update the countdown timer
        elapsed_time = time.time() - last_score_time
        remaining_time = countdown_time - elapsed_time

        # If the countdown reaches zero, end the game
        # If the countdown reaches zero, end the game
        if remaining_time <= 0:
            game_running = False  # End the game if no score in 2 seconds
            game_over = True  # Set the game_over flag to True

        i = 0
        while i < len(trackers):
            tracker = trackers[i]
            success, bbox = tracker.update(frame)
            if success:
                (x, y, w, h) = [int(v) for v in bbox]
                trackers_boxes[i] = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                if (cursor_x <= x + w and cursor_x + cursor_size >= x) and (cursor_y <= y + h and cursor_y + cursor_size >= y):
                    score += 1
                    trackers.pop(i)
                    trackers_boxes.pop(i)
                    last_score_time = time.time()  # Reset the score timer
                    countdown_time = 2  # Reset countdown for the next score
                    continue

                for square in squares:
                    if (square['x'] <= x + w and square['x'] + square['size'] >= x) and (square['y'] <= y + h and square['y'] + square['size'] >= y):
                        score += 1
                        trackers.pop(i)
                        trackers_boxes.pop(i)
                        break
            else:
                trackers.pop(i)
                trackers_boxes.pop(i)
                continue
            i += 1

        # Draw additional squares
        for square in squares:
            fill_color = (0, 0, 255)  # Red
            cv2.rectangle(frame, 
                          (square['x'], square['y']), 
                          (square['x'] + square['size'], square['y'] + square['size']), 
                          fill_color, cv2.FILLED)

        # Display score and countdown timer
        draw_text(frame, f"Score: {score}", (10, 80), font_scale=2)
        draw_text(frame, f"Time: {remaining_time:.2f} sec", (10, 130), font_scale=1.5, color=(0, 255, 0))  # Display countdown

        cv2.imshow("CatchYa!", frame)

        # Controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            game_running = False
        elif key == ord('a') and cursor_x > 0:
            cursor_x -= 10
        elif key == ord('d') and cursor_x < WIDTH - cursor_size:
            cursor_x += 10
        elif key == ord('w') and cursor_y > 0:
            cursor_y -= 10
        elif key == ord('s') and cursor_y < HEIGHT - cursor_size:
            cursor_y += 10
        elif key == ord('p'):
            paused = not paused

    elif game_over:
        game_over_screen(score)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            # Reset variables for restarting the game
            game_running = True
            game_over = False
            paused = False
            score = 0
            cursor_x = WIDTH // 2
            cursor_y = HEIGHT // 2
            start_time = time.time()
            last_score_time = start_time
            trackers = []  # Clear all trackers
            trackers_boxes = []  # Clear all tracker boxes
        elif key == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()
