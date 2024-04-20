import cv2
import sys

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if __name__ == '__main__':

    # Set up tracker

    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', "CRST"]
    tracker_type = tracker_types[2]

    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    if tracker_type == "CRST":
        tracker = cv2.TrackerCSRT_create()

    video = cv2.VideoCapture("video.mp4")

    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mp4 = cv2.VideoWriter_fourcc(*'MP4V')
    output = cv2.VideoWriter("test.mov", mp4, 25, (frame_width, frame_height))

    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    cv2.namedWindow("roi", cv2.WINDOW_NORMAL)
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    # Define an initial bounding box
    bbox = (287, 23, 86, 320)

    bbox = cv2.selectROI("roi", frame, False)

    ok = tracker.init(frame, bbox)

    while True:
        ok, frame = video.read()
        if not ok:
            break

        timer = cv2.getTickCount()
        ok, bbox = tracker.update(frame)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Drop bounding box
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
        else:
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
        cv2.imshow("Tracking", frame)
        output.write(frame)

        # Exit if ESC pressed
        t = cv2.waitKey(1) & 0xff
        if t == 27:
            break