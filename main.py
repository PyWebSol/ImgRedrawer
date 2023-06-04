import cv2
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

import shutil
import os
import glob

import random

Status = 1

symbols = list('abcdefghijkmnopqrstuvwxyzABCDEFGHIJKMNOPQRSTUVWXYZ1234567890')
outfname = ''

workmode = input('Какие настройки вы хотите использовать?\n \n1) Минимальные\n2) Средние\n3) Средние +\n4) Ультра\n5) Ультра +\n6) Свои настройки\n \nВведите ответ (1/2/3/4/5/6): ')

descriptions = {1: 'Минимальные', 
                2: 'Средние', 
                3: 'Средние +', 
                4: 'Ультра', 
                5: 'Ультра +', 
                6: 'Свои настройки'}

while Status == 1:
    if workmode.isdigit() and int(workmode) > 0 and int(workmode) <= 6:
        Status = 0
        workmode = int(workmode)
        if workmode == 1:
            print(f'Выбран режим {workmode} ({descriptions[workmode]})')
            neurons = 64
            slices = 10
            epochs4set = 2
            division = 3
            resdivision = 1 / 1.2
        elif workmode == 2:
            print(f'Выбран режим {workmode} ({descriptions[workmode]})')
            neurons = 128
            slices = 10
            epochs4set = 3
            division = 2
            resdivision = 1 / 1.2
        elif workmode == 3:
            print(f'Выбран режим {workmode} ({descriptions[workmode]})')
            neurons = 128
            slices = 15
            epochs4set = 4
            division = 2
            resdivision = 1 / 1.2
        elif workmode == 4:
            print(f'Выбран режим {workmode} ({descriptions[workmode]})')
            neurons = 128
            slices = 20
            epochs4set = 10
            division = 2
            resdivision = 1 / 1.2
        elif workmode == 5:
            print(f'Выбран режим {workmode} ({descriptions[workmode]})')
            neurons = 128
            slices = 20
            epochs4set = 10
            division = 1
            resdivision = 1 / 1.2
        elif workmode == 6:
            print(f'Выбран режим {workmode} ({descriptions[workmode]})')
            neurons = 128
            slices = 20
            epochs4set = 10
            division = 1
            resdivision = 1 / 1.2
    else:
        workmode = input('Введите верное значение!\n:')

project_directory = os.getcwd()

while True:
    mod3 = input('Выберите тип файла:\n \n1) Фото\n2) Видео\n: ')
    if mod3.isdigit() and int(mod3) > 0 and int(mod3) <= 2:
        mod3 = int(mod3)
        break
    else:
        print('Введите верное число!\n: ')

class GetFiles:
    def get_jpg_files():
        jpg_files = glob.glob(project_directory + "/*.jpg")
        return jpg_files

    def get_mp4_files():
        mp4_files = glob.glob(project_directory + "/*.mp4")
        return mp4_files

if mod3 == 1:
    ffiles = GetFiles.get_jpg_files()
    for iterat in range(5):
        outfname += random.choice(symbols)
    outfname += '.jpg'
else:
    ffiles = GetFiles.get_mp4_files()
    for iterat in range(5):
        outfname += random.choice(symbols)
    outfname += '.mp4'

msg = f'Выберите путь к файлу:\n \n'

filesnamesandnums = {}

alliters = []

if not len(ffiles) == 0:
    for fpath, iteraion in zip(ffiles, range(len(ffiles))):
        msg += f'{iteraion + 1}) {fpath}\n'
        filesnamesandnums[iteraion + 1] = fpath
        alliters.append(iteraion + 1)
else:
    print('Файлы с данным форматом не найдены!')
    exit(0)

print(msg)
fnum = input(': ')
if fnum.isdigit() and int(fnum) in alliters:
    fpath = filesnamesandnums[int(fnum)]
else:
    print('Введите верное значение (номер файла)!')

class video_redraw:
    def split_video(video_path, output_folder):
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            print("Не удалось открыть видео.")
            return
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        frame_count = 0

        while True:
            ret, frame = video.read()
            if not ret:
                break
            frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
            frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            if frame_count % division == 0:
                cv2.imwrite(frame_path, frame)
                print(f'Кадр номер {round((frame_count + 1) / division)} успешно сохранен', end='\r')
            frame_count += 1
        video.release()
        print(f"\nВидео разделено на {round(frame_count / division)} кадров.")

    def rmdirs(dir1, dir2):
        if os.path.exists(dir1):
            shutil.rmtree(dir1)
        if os.path.exists(dir2):
            shutil.rmtree(dir2)

    def redrawimg(path):
        mdel = [Dense(16, activation='relu', input_shape=(2,))]
        for i in range(slices):
            mdel.append(Dense(neurons, activation='relu'))
        mdel.append(Dense(3, activation='sigmoid'))
        model = Sequential(mdel)
        image_path = path
        image = Image.open(image_path)
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        input_shape = image_array.shape[:2]
        input_data = []
        target_data = []
        for x in range(input_shape[0]):
            for y in range(input_shape[1]):
                input_data.append([x / input_shape[0], y / input_shape[1]])
                target_data.append(image_array[x, y] / 255.0)
        input_data = tf.convert_to_tensor(input_data, dtype=tf.float32)
        target_data = tf.convert_to_tensor(target_data, dtype=tf.float32)
        model.compile(optimizer='adamax', loss='mean_squared_error', )
        model.fit(input_data, target_data, epochs=epochs4set)
        output_data = model.predict(input_data) * 255.0
        output_array = output_data.reshape(input_shape + (3,))
        output_image = Image.fromarray(output_array.astype('uint8'), mode='RGB')
        draw = ImageDraw.Draw(output_image)
        watermark_text = "@PyWebSol"
        font_size = int(output_image.width / 20)
        font = ImageFont.truetype("arial.ttf", font_size)
        text_bbox = draw.textbbox((0, 0), watermark_text, font=font)
        text_position = (output_image.width // 2 - text_bbox[2] // 2, output_image.height - text_bbox[3] - 10)
        draw.text(text_position, watermark_text, font=font, fill=(255, 255, 255, 15))
        return output_image

    def redraw(input_folder, output):
        if not os.path.exists(output):
            os.makedirs(output)
        pathes = []
        unsorted = []
        iters = []
        for filename, iteration in zip(os.listdir(input_folder), range(len(os.listdir(input_folder)))):
            if filename.endswith(".jpg"):
                frame_path = os.path.join(input_folder, filename)
                unsorted.append(frame_path)
                iters.append(iteration)
        pathes = sorted(unsorted, key=lambda x: int(x.split('_')[1].split('.')[0]))
        for fpath, iteration in zip(pathes, iters):
            video_redraw.redrawimg(fpath).save(f"{output}/frame_{iteration}.jpg")
            print(f'Кадр номер {iteration + 1} успешно перерисован')

    def merge_frames(input_folder, output_video, fps):
        frames = []
        pathes = []
        unsorted = []
        iters = []
        for filename in os.listdir(input_folder):
            if filename.endswith(".jpg"):
                frame_path = os.path.join(input_folder, filename)
                unsorted.append(frame_path)
        pathes = sorted(unsorted, key=lambda x: int(x.split('_')[1].split('.')[0]))
        for i in pathes:
            frame = cv2.imread(i)
            frames.append(frame)
        height, width, channels = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        for frame in frames:
            video_writer.write(frame)
        video_writer.release()
        print(f"Кадры объединены в видео: {output_video}")
    
    def get_video_fps(video_path):
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        video.release()
        return fps / division
    
    def end():
        print('Перерисовка видео завершена!')

class photo_redraw:
    def redraw(path, out):
        mdel = [Dense(16, activation='relu', input_shape=(2,))]
        for i in range(slices):
            mdel.append(Dense(neurons, activation='relu'))
        mdel.append(Dense(3, activation='sigmoid'))
        model = Sequential(mdel)
        image_path = path
        image = Image.open(image_path)
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        input_shape = image_array.shape[:2]
        input_data = []
        target_data = []

        for x in range(input_shape[0]):
            for y in range(input_shape[1]):
                input_data.append([x / input_shape[0], y / input_shape[1]])
                target_data.append(image_array[x, y] / 255.0)

        input_data = tf.convert_to_tensor(input_data, dtype=tf.float32)
        target_data = tf.convert_to_tensor(target_data, dtype=tf.float32)
        model.compile(optimizer='adamax', loss='mean_squared_error', )
        model.fit(input_data, target_data, epochs=epochs4set)
        output_data = model.predict(input_data) * 255.0
        output_array = output_data.reshape(input_shape + (3,))
        output_image = Image.fromarray(output_array.astype('uint8'), mode='RGB')
        draw = ImageDraw.Draw(output_image)
        watermark_text = "@PyWebSol"
        font_size = int(output_image.width / 20)
        font = ImageFont.truetype("arial.ttf", font_size)
        text_bbox = draw.textbbox((0, 0), watermark_text, font=font)
        text_position = (output_image.width // 2 - text_bbox[2] // 2, output_image.height - text_bbox[3] - 10)
        draw.text(text_position, watermark_text, font=font, fill=(255, 255, 255, 15))
        output_image.save(out)
    
    def end():
        print(f'Изображение было сохранено как {output_photo_path}')
    

if mod3 == 1:
    photo_path = fpath
    output_photo_path = f'photos/{outfname}'
    if not os.path.exists('photos'):
            os.makedirs('photos')
    photo_redraw.redraw(photo_path, output_photo_path)
    photo_redraw.end()
    
elif mod3 == 2:
    video_path = fpath
    output_folder = "frames"
    out2neuro = "redrawed"
    output_video = f'videos/{outfname}'
    if not os.path.exists('videos'):
            os.makedirs('videos')
    fps = video_redraw.get_video_fps(video_path)
    video_redraw.split_video(video_path, output_folder)
    video_redraw.redraw(output_folder, out2neuro)
    video_redraw.merge_frames(out2neuro, output_video, fps)
    video_redraw.rmdirs(output_folder, out2neuro)
    video_redraw.end()