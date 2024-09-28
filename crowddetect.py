import cv2
from ultralytics import YOLO

model = YOLO('yolov8m.pt') 

# Function to process frames and detect people
def process_frame(frame, crowd_threshold=8):
    # Perform detection (returns a list of detected objects with bounding boxes and class labels)
    results = model(frame)

    # Initialize the counter for the number of people detected
    people_count = 0

    # Loop through the results and draw bounding boxes for people
    for result in results:
        for box in result.boxes:
            # Check if the detected object is a person (COCO class id for person is 0)
            if box.cls == 0:  # Class 0 corresponds to 'person'
                people_count += 1
                # Get the coordinates of the bounding box (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()

                # Draw the bounding box around the person
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Optionally, add a label above the box
                label = f"Person {people_count}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the count of people on the screen
    cv2.putText(frame, f"People Count: {people_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Check if the crowd exceeds the threshold
    if people_count > crowd_threshold:
        overcrowded_alert = "Alert: Overcrowded!"
        cv2.putText(frame, overcrowded_alert, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    return frame

# Function to process a video file
def process_video(input_video_path, output_video_path, crowd_threshold=8):
    # Open the input video file
    cap = cv2.VideoCapture(input_video_path)
    
    # Get the width, height, and frames per second (FPS) of the video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create a VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi files
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Process each frame from the input video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame (detect people and draw bounding boxes)
        processed_frame = process_frame(frame, crowd_threshold)

        # Write the processed frame to the output video
        out.write(processed_frame)

        # Optionally, display the frame (for debugging)
        # cv2.imshow("Processed Video", processed_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Release the video objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Function to process real-time webcam feed
def process_webcam(crowd_threshold=3):
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the current frame
        processed_frame = process_frame(frame, crowd_threshold)

        # Display the processed frame
        cv2.imshow("Crowd Management - YOLOv8", processed_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    # process_video(r'C:\Users\snsis\Desktop\Proj\cctvdetect\cctvcrime\input _video.mp4', r'C:\Users\snsis\Desktop\Proj\cctvdetect\cctvcrime\output_vid.mp4')

    process_webcam()
