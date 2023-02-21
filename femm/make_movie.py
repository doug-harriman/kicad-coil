def make_movie(DirName, MovieFileName):
    # =============================================================================
    #     import os
    #     import moviepy.video.io.ImageSequenceClip
    # # see https://stackoverflow.com/questions/44947505/how-to-make-a-movie-out-of-images-in-python
    #     fps=1
    #
    #     image_files = [DirName+'/'+img for img in os.listdir(DirName) if img.endswith(".png")]
    #     clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    #     clip.write_videofile(MovieFileName)
    #
    # =============================================================================
    import cv2
    import os

    image_folder = DirName
    video_name = MovieFileName

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Documentation:
    # https://docs.opencv.org/4.x/dd/d9e/classcv_1_1VideoWriter.html#ad59c61d8881ba2b2da22cff5487465b5
    format = 0  # Zero compression, just appended full images.
    format = cv2.VideoWriter_fourcc('M', 'P', '4', '2')  # MPEG-4

    video = cv2.VideoWriter(video_name,
                            format,  # 'fourcc' - Compression info.
                            5,  # Frame rate in frames/sec
                            (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


if __name__ == "__main__":

    make_movie('.', 'test.avi')
