import cv2
import numpy as np
from matplotlib import pyplot as plt

# Definir función para rotar la imagen
def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
    return rotated

def method_1():
    # Código principal del programa
    print("Test\n")
    img_rgb = cv2.imread('images/label.jpeg')
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('images/s.png', 0)
    h, w = template.shape[::] 
    res = cv2.matchTemplate(img_gray, template, cv2.TM_SQDIFF)

    # For TM_SQDIFF, Good match yields minimum value; bad match yields large values
    # For all others it is exactly opposite, max value = good fit.
    plt.imshow(res, cmap='gray')

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print(min_loc)

    top_left = min_loc  #Change to max_loc for all except for TM_SQDIFF
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img_gray, top_left, bottom_right, 255, 2)  #White rectangle with thickness 2. 

    cv2.imshow("Matched image", img_gray)
    cv2.waitKey()
    cv2.destroyAllWindows()

def method_2():  
    img_rgb = cv2.imread('images/label.jpg')
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('images/s.jpg',0)
    h, w = template.shape[::]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    plt.imshow(res, cmap='gray')

    threshold = 0.9 #Pick only values above 0.8. For TM_CCOEFF_NORMED, larger values = good fit.

    loc = np.where( res >= threshold)  
    print(loc)
    #Outputs 2 arrays. Combine these arrays to get x,y coordinates - take x from one array and y from the other.

    #Reminder: ZIP function is an iterator of tuples where first item in each iterator is paired together,
    #then the second item and then third, etc. 

    for pt in zip(*loc[::-1]):   #-1 to swap the values as we assign x and y coordinate to draw the rectangle. 
        #Draw rectangle around each object. We know the top left (pt), draw rectangle to match the size of the template image.
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 1)  #Red rectangles with thickness 2. 

    #cv2.imwrite('images/template_matched.jpg', img_rgb)
    cv2.imshow("Matched image", img_rgb)
    cv2.waitKey()
    cv2.destroyAllWindows()  

def mejor_rotacion(img,img_template):
    # Cargar imagen y template
    image = cv2.imread(img, 0)
    template = cv2.imread(img_template, 0)
    # Intentar con diferentes rotaciones
    best_match = None
    best_val = -np.inf
    best_angle = 0
    for angle in range(0, 360, 15):  # Rotar en pasos de 15 grados
        rotated_template = rotate_image(template, angle)
        
        # Aplicar matchTemplate
        result = cv2.matchTemplate(image, rotated_template, cv2.TM_CCOEFF_NORMED)
        
        # Encontrar el valor máximo de coincidencia
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val > best_val:  # Si el resultado es mejor, lo guardamos
            best_val = max_val
            best_match = max_loc
            best_angle = angle
    return best_angle

def method_3():
    img_rgb = cv2.imread('images/f16.jpg') #Large image
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('images/f16_template.jpg', 0)  #Small image (template)

    import imutils
    # Clculate the metric for varying image sizes
    #pick the one that gives the best metric (e.g. Minimum Sq Diff.)

    best_match = None
    for scale in np.linspace(0.055, 0.5, 11):  #Pick scale based on your estimate of template to object in the image ratio
        print(scale)

        
    #Resize the input template image
        resized_template = imutils.resize(template, width = int(template.shape[1] * scale))
        
        res = cv2.matchTemplate(img_gray, resized_template, cv2.TM_SQDIFF)
        min_val, _, min_loc, _ = cv2.minMaxLoc(res)  #Only care about minimum value and location as we are using TM_SQDIFF
        
        #Check if the min_val is the minimum compared to the value from other scales templates
        #If it is minimum then we got a better match compared to other scales
        #So save the value and location. 
        if best_match is None or min_val <= best_match[0]:
            ideal_scale=scale  #Save the ideal scale for printout. 
            h, w = resized_template.shape[::] #Get the size of the scaled template to draw the rectangle. 
            best_match = [min_val, min_loc, ideal_scale]
            
            
    print("Ideal template image size is : ", int(template.shape[0]*ideal_scale), "x", int(template.shape[1]*ideal_scale))

    #Save the image with a red box around the detected object in the large image. 
    top_left = best_match[1]  #Change to max_loc for all except for TM_SQDIFF
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img_rgb, top_left, bottom_right, (0, 0,255), 2)  #Red rectangle with thickness 2. 
    cv2.imwrite('matched_resized.jpg', img_rgb)

def method_4():
    # Cargar imagen y template
    #image = cv2.imread('images/label.jpg', 0)
    #template = cv2.imread('images/s.jpg', 0)
    
    image = cv2.imread('images/f16.jpg', 0)
    template = cv2.imread('images/f16_template.jpg', 0)
        
    # Crear el detector ORB
    orb = cv2.ORB_create()

    # Detectar keypoints (puntos clave) y calcular descriptores para el template y la imagen
    kp1, des1 = orb.detectAndCompute(template, None)  # Para el template
    kp2, des2 = orb.detectAndCompute(image, None)     # Para la imagen

    # Usar BFMatcher para comparar los descriptores
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Ordenar las coincidencias según la distancia
    matches = sorted(matches, key=lambda x: x.distance)

    # Obtener las coordenadas del mejor match (menor distancia)
    if matches:
        best_match = matches[0]
        img_idx = best_match.trainIdx
        template_idx = best_match.queryIdx

        # Obtener el keypoint de la imagen y del template
        (x1, y1) = kp2[img_idx].pt  # Coordenadas en la imagen
        (x2, y2) = kp1[template_idx].pt  # Coordenadas en el template (para referencia)

        # Obtener las dimensiones del template
        h, w = template.shape[:2]

        # Dibujar el rectángulo en la imagen que rodea el área encontrada
        top_left = (int(x1 - w // 2), int(y1 - h // 2))
        bottom_right = (int(x1 + w // 2), int(y1 + h // 2))
        img_with_rectangle = cv2.rectangle(image.copy(), top_left, bottom_right, (0, 255, 0), 3)  # Color verde

        # Mostrar la imagen con el rectángulo
        cv2.imshow('Coincidencia Encontrada', img_with_rectangle)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No se encontraron coincidencias.")
  
    
def main():
    #method_1()
    method_2()
    #method_3()
    #method_4()

    


# Verificar si el script se está ejecutando directamente
if __name__ == "__main__":
    main()