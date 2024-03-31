from fastapi import FastAPI, WebSocket
import asyncio
import glob
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
from scipy.optimize import linear_sum_assignment
from track_12 import country_balls_amount, track_data
import cv2
import pickle

selected_tracks = 'track_12'


app = FastAPI(title='Tracker assignment')
imgs = glob.glob('imgs/*')
country_balls = [{'cb_id': x, 'img': imgs[x % len(imgs)]} for x in range(country_balls_amount)]
print('Started')


def euclidean_distance_2d(x, y):
    xx = np.array(x)
    yy = np.array(y)
    return np.sqrt(np.sum((xx - yy) ** 2, axis=-1))


def get_distance_matrix(prev_centers, current_centers):
    prev_centers_array = np.array(prev_centers)
    current_centers_array = np.array(current_centers)

    distance_matrix = np.zeros((len(prev_centers), len(current_centers)))

    for i, prev_center in enumerate(prev_centers_array):
        for j, current_center in enumerate(current_centers_array):
            distance_matrix[i, j] = euclidean_distance_2d(prev_center, current_center)

    return distance_matrix


def to_opencv_boxes(xyxy_bbox):
    x_left_upper, y_left_upper, x_right_bottom, y_right_bottom = xyxy_bbox
    w, h = (x_right_bottom - x_left_upper), (y_right_bottom - y_left_upper)
    return [x_left_upper, y_left_upper, w, h]


def to_yolo_boxes(xyxy_bbox):
    x_left_upper, y_left_upper, x_right_bottom, y_right_bottom = xyxy_bbox
    x_c, y_c = (x_left_upper + x_right_bottom) / 2, (y_left_upper + y_right_bottom)/2
    w, h = (x_left_upper - x_right_bottom), (y_left_upper - y_right_bottom)
    return [x_c, y_c, w, h]

free_ids = [x for x in range(100)]


def tracker_soft(el, prev_el):

    if not prev_el:
        for idx, x in enumerate(el['data']):
            if x['bounding_box']:
                x['track_id'] = free_ids.pop(0)

            else:
                x['track_id'] = None
        return el

    for idx, x in enumerate(el['data']):
        if not x['bounding_box']:
            x['track_id'] = None

    for idx, x in enumerate(prev_el['data']):
        if not x['bounding_box']:
            x['track_id'] = None

    prev_elements = [to_yolo_boxes(detection['bounding_box'])[:2] for detection in prev_el['data'] if detection['bounding_box']]
    current_elements = [to_yolo_boxes(detection['bounding_box'])[:2] for detection in el['data'] if detection['bounding_box']]

    distance_matrix = get_distance_matrix(prev_elements, current_elements)
    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    for row, col in zip(row_ind, col_ind):
        el['data'][col]['track_id'] = prev_el['data'][row]['track_id']

    for idx, x in enumerate(el['data']):
        if x['bounding_box'] and x['track_id'] is None:
            x['track_id'] = free_ids.pop(0)

    taken_ids = [x['track_id'] for x in el['data'] if x['bounding_box']]
    total_ids = [i for i in range(100)]

    if len(current_elements) < len(prev_elements):
        for x in total_ids:
            if x not in taken_ids and x not in free_ids:
                free_ids.append(x)

    free_ids.sort()

    return el


def tracker_strong(el, tracker):

    i = 0
    for idx, x in enumerate(el['data']):
        if x['bounding_box']:
            x['track_id'] = i
            i += 1
        else:
            x['track_id'] = None

    bboxes = []
    path_to_frame = './frames/'+ selected_tracks + '/' + str(el['frame_id']) + '.png'
    im = cv2.imread(path_to_frame)
    frame = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    if im is None or frame.size == 0:
        raise ValueError(f"Frame is empty or not loaded correctly from {path_to_frame}")

    for idx, x in enumerate(el['data']):
        if x['bounding_box']:
            bboxes.append((to_opencv_boxes(x['bounding_box']), 1, 1))  # [opencv box], confidence, class

    tracks = tracker.update_tracks(bboxes, frame=frame)

    bboxes_tracks = {track.track_id: to_yolo_boxes(track.to_ltrb())[:2] for track in tracks}

    for idx, x in enumerate(el['data']):
        if x['bounding_box']:
            if bboxes_tracks:
                bbox_center = to_yolo_boxes(x['bounding_box'])[:2]
                track_id, _ = min(bboxes_tracks.items(), key=lambda item: euclidean_distance_2d(item[1], bbox_center))

                x['track_id'] = track_id

    return el




@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print('Accepting client connection...')
    await websocket.accept()
    # отправка служебной информации для инициализации объектов
    # класса CountryBall на фронте
    tracker = DeepSort(max_age=5)
    ids_dict = {ii : [] for ii in range(country_balls_amount)}
    await websocket.send_text(str(country_balls))
    prev_el = []

    for el in track_data:
        await asyncio.sleep(0.5)
        # TODO: part 1
        # tmp_el = tracker_soft(el, prev_el)
        # prev_el = el
        # el = tmp_el


        el = tracker_strong(el, tracker)

        for idx, x in enumerate(el['data']):
            if x['track_id'] != None:
                ids_dict[x['cb_id']].append(x['track_id'])

        # TODO: part 2
        # отправка информации по фрейму
        await websocket.send_json(el)



    print(ids_dict)
    with open('./metrics_dicts/track_12_strong_ids.pkl', 'wb') as f:
        pickle.dump(ids_dict, f)
    print('Bye..')
