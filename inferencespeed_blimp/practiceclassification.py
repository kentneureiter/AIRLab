import sensor
import time

# Camera setup
sensor.reset()
sensor.ioctl(sensor.IOCTL_SET_FOV_WIDE, True)
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.HQVGA)
sensor.skip_frames(time=2000)

clock = time.clock()

# LAB thresholds - tuned to your hardware measurements
A_Orange_Min = 25
A_Orange_Max = 35
B_Orange_Min = 25
B_Orange_Max = 35

A_Blue_Min = 14
B_Blue_Max = -40

# Grid setup
N_rows = 14
N_columns = 21
cell_width = 240 // N_columns
cell_height = 160 // N_rows

# Frame center for offset calculation
frame_cx = 240 // 2
frame_cy = 160 // 2

while True:
    clock.tick()
    img = sensor.snapshot()
    # Tracking variables for each color
    orange_sum_x, orange_sum_y, orange_count = 0, 0, 0
    blue_sum_x, blue_sum_y, blue_count = 0, 0, 0
    start = time.ticks_us()
    for r in range(N_rows):
        for c in range(N_columns):
            roi = (c * cell_width, r * cell_height, cell_width, cell_height)
            stats = img.get_statistics(roi=roi)
            a = stats.a_mean()
            b = stats.b_mean()

            # Cell center in pixel coordinates
            cell_cx = c * cell_width + cell_width // 2
            cell_cy = r * cell_height + cell_height // 2

            if A_Orange_Min < a < A_Orange_Max and B_Orange_Min < b < B_Orange_Max:
                # Orange cell detected
                img.draw_rectangle(roi, color=(255, 100, 0))
                orange_sum_x += cell_cx
                orange_sum_y += cell_cy
                orange_count += 1

            elif a > A_Blue_Min and b < B_Blue_Max:
                # Blue cell detected
                img.draw_rectangle(roi, color=(0, 0, 255))
                blue_sum_x += cell_cx
                blue_sum_y += cell_cy
                blue_count += 1
    end = time.ticks_us()
    inference_ms = time.ticks_diff(end, start) / 1000
    print("Inference ms:", inference_ms)
    # Compute centroids and offset from frame center
    if orange_count > 0:
        orange_x = orange_sum_x // orange_count
        orange_y = orange_sum_y // orange_count
        offset_x = orange_x - frame_cx
        offset_y = orange_y - frame_cy
        img.draw_circle(orange_x, orange_y, 8, color=(255, 100, 0), thickness=2)
        print("ORANGE at:", orange_x, orange_y,
              "| offset x:", offset_x, "y:", offset_y,
              "| cells:", orange_count)

    if blue_count > 0:
        blue_x = blue_sum_x // blue_count
        blue_y = blue_sum_y // blue_count
        offset_x = blue_x - frame_cx
        offset_y = blue_y - frame_cy
        img.draw_circle(blue_x, blue_y, 8, color=(0, 0, 255), thickness=2)
        print("BLUE at:", blue_x, blue_y,
              "| offset x:", offset_x, "y:", offset_y,
              "| cells:", blue_count)

    if orange_count == 0 and blue_count == 0:
        print("No target detected | FPS:", round(clock.fps(), 1))
"""
import sensor
import time
sensor.reset()
sensor.ioctl(sensor.IOCTL_SET_FOV_WIDE, True)
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.HQVGA)
sensor.skip_frames(time=2000)
clock = time.clock()
A_Orange_Min = 25
A_Orange_Max = 35
B_Orange_Min = 25
B_Orange_Max = 35
A_Blue_Min = 14
B_Blue_Max = -40
N_rows = 14
N_columns = 21
cell_width = 240//N_columns
cell_height = 160//N_rows
while True:
    clock.tick()
    img = sensor.snapshot()
    best_x, best_y = 0, 0
    best_score = 0
    for r in range(N_rows):
        for c in range(N_columns):
            roi = (c*cell_width, r*cell_height, cell_width, cell_height)
            stats = img.get_statistics(roi=roi)
            a = stats.a_mean()
            b = stats.b_mean()
            # simple score - how red is this cell?
            score = a - abs(b)
            if a > A_MIN and b > B_MIN:
                img.draw_rectangle(roi, color=(255, 0, 0))
                if score > best_score:
                    best_score = score
                    best_x = c*cell_width + cell_width//2
                    best_y = r*cell_height + cell_height//2
    if best_score > 0:
        img.draw_circle(best_x, best_y, 10, color=(0, 255, 0), thickness=2)
        print("Target at:", best_x, best_y, "Score:", round(best_score, 1))
"""





