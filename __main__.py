import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

# images path 
video_path = 'Airplane.mp4'
out_path = 'mean_shift.mp4'

# video capture
cap = cv2.VideoCapture(video_path)

# check video ok
if (cap.isOpened() == False): 
  print("Error en leer el video")

# get video info
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(5))
total_frames = int(cap.get(7))

# proggres bar           
pbar = tqdm(total= total_frames, desc='Video processing')

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_width,frame_height))

# run all frames
for frame_id in range(total_frames):
  
  # read new frame
  ret, frame = cap.read()

  if ret == True: 
    
    # if it is the first frame
    if frame_id == 0:
        
        # roi with the plane position
        ROI = (680, 220, 50, 120)
        
        # plane cropped
        plane = frame.copy()[ROI[1]: ROI[1] +ROI[3], ROI[0]: ROI[0] + ROI[2]]
        
        # create a mask
        ret, bkg_mask = cv2.threshold(cv2.cvtColor(plane, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # calc of the histogram
        histObject = cv2.calcHist([plane], [0], bkg_mask, [256], [0,256])  

        # plot roi
        cv2.rectangle(frame,  (ROI[0], ROI[1]), (ROI[0]+ ROI[2], ROI[1] +ROI[3]),  (255, 0, 0), 2)

        # show the results
        plt.title('ROI first frame')
        plt.imshow(frame)
        plt.show()
        plt.title('Plane')
        plt.imshow(plane)
        plt.show()
        plt.title('Mask')
        plt.imshow(bkg_mask, cmap = 'gray')
        plt.show()


    else:
        # Setup the termination criteria, either 10 iterations or move by atleast 1 pt
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

        # background projection calc
        backProjectImage = cv2.calcBackProject([frame], [0], histObject, [0,255], 1)

        # Compute the new window using mean shift in the present frame
        ret, ROI = cv2.meanShift(backProjectImage, ROI, term_crit)
        
        # plot roi
        cv2.rectangle(frame,  (ROI[0], ROI[1]), (ROI[0]+ ROI[2], ROI[1] +ROI[3]),  (255, 0, 0), 2)


    # write output video
    out.write(frame)

    # update progress bar
    pbar.update(1)

  # exit 
  else:
    break  

# release
cap.release()
out.release()

# close progress bar
pbar.close()