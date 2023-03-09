import numpy as np
import cv2 as cv2

#kép beolvasás
atkelo = cv2.imread("crosswalk.jpg")
cv2.imshow('eredeti', atkelo)

#szürkeárnyalatos konverzió
grayCrosswalk = cv2.cvtColor(atkelo, cv2.COLOR_BGR2GRAY)
cv2.imshow('szurkearnyalatos', grayCrosswalk)

#Gauss-simított kép
szurkeSimitottAtkelo = cv2.GaussianBlur(grayCrosswalk, (5, 5), 0)
cv2.imshow('szurkeSimitottAtkelo', szurkeSimitottAtkelo)

#szintervaltas a grayCrosswalkBlurred keppen
szinterValtottGCB = cv2.cvtColor(szurkeSimitottAtkelo, cv2.COLOR_BGR2RGB)
cv2.imshow('szinter valtott szurkeSimitottAtkelo', szinterValtottGCB)

#körvonal keresés
cannyKorvonalasKep = cv2.Canny(szinterValtottGCB, 40, 260)
cv2.imshow('canny - korvonalas kep', cannyKorvonalasKep)

#Morfológiai szűrés - megvastagított körvonal
strukturaloEllipszis = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

dilate = cv2.dilate(cannyKorvonalasKep, strukturaloEllipszis, iterations=1)
cv2.imshow('dilate', dilate)

#kontur kereses
konturok, _  = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#konturok teruletének kiszamola majd azok kitöltése piros szinnel
for kontur in konturok:
    terulet = cv2.contourArea(kontur)
    if terulet > 1000:
         cv2.drawContours(atkelo, [kontur], -1, (0, 0, 255), -1)

cv2.imshow('pirosraFestettAtkelo', atkelo)
cv2.imwrite('result.jpg', atkelo)

cv2.waitKey(0)