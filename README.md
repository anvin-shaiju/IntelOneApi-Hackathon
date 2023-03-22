### IntelOneApi-Hackathon
# Intel OneApi Hackathon- Team "DEBUG THUGS"-Smart Belt For Spinal Diseases

![image](https://user-images.githubusercontent.com/113662146/226708692-e0bafd9b-0712-4c02-a875-4214c27ca899.png)


### TOPIC: SmartBelt With Android Access(Intel One-API Tools) For LUMBAR SPONDYLOSIS & all common back and spinal problems

We have created a Wearable belt ,allowing for real-time monitoring of the patient's spinal activity and motions. An Android or iOS app will be used to gather, analyze, and send this data to the patient, the doctor, and any relevant family members. It displays details about the bend angle, strain, and pressure applied. provide specific stretching and exercise routines to improve and treat their ailment. By consistently keeping an eye on things and giving feedback, we can help stop the spine from suffering more harm while also promoting healing and recovery.If the patient has been dealing with significant pain and movement abnormalities, the devices implanted inside the belt can transfer vibrations or electric impulses to the spines. it functions as treatment.

![image](https://user-images.githubusercontent.com/113662146/226709926-1b960a3c-57e0-4e46-a350-11fb76b81f74.png)


### Technologies Used:

![image](https://user-images.githubusercontent.com/113662146/226709852-51ac94fc-8ca4-4eb6-9328-57f464b3bf58.png)
![image](https://user-images.githubusercontent.com/113662146/226709420-45ca713e-f7e5-466b-a4b9-8a55e525b18d.png)
![image](https://user-images.githubusercontent.com/113662146/226710569-d04e70cc-b496-43ae-8363-ffd72e301d79.png)
![image](https://user-images.githubusercontent.com/113662146/226710315-61535777-8ae6-4909-9213-2c5832768240.png)


#### RASPBERRY PI
#### INTEL ONEAPI TOOLKITS(Math kernel library,AI analytics toolkit,and Basic Python toolkit)
#### Math kernel library- Mathematical Calculations for average values of the sensor
#### AI analytics toolkit- For Data Analysis and Finding the Accurate rate of movements and remedies

##### //Towards Advanced Level we have to use-Intel® oneAPI Deep Neural Networks Library & Intel® oneAPI Collective Communications Library

![image](https://user-images.githubusercontent.com/113662146/226713324-d1f659fa-69f3-4256-b6e3-da6d1cbad99a.png)

#### The solution for the problem has been found and we have made teh protoype,Now if we take it to advanced stages the application will be numerous
![image](https://user-images.githubusercontent.com/113662146/226716356-f075e7fc-b801-4608-8f60-b41ea8ed434d.png)

#### The statistics representing the need for the project in a picture
![image](https://user-images.githubusercontent.com/113662146/226716649-837d2930-4768-4095-b1c6-9bbff09613ae.png)


## Steps to Follow:

#### Connect flex Sensor to Raspberry Pi Using an extension Board attached to raspberry Pi(With proper circuits) 
 
#### Install Pyserial Library using the following command
![image](https://user-images.githubusercontent.com/113662146/226814998-9476c6da-e1a9-47ea-bd54-fbddf19ba34c.png)

#### Paste This code snippet(LOGIC) in the raspberry Pi environment that You are running
Full code to be accessed in Github Repositry

while True:
    # Read analog value 
    GPIO.output(18, GPIO.HIGH)
    time.sleep(0.000002)
    GPIO.output(18, GPIO.LOW)
    while GPIO.input(24) == GPIO.LOW:
        pass
    start = time.time()
    while GPIO.input(24) == GPIO.HIGH:
        pass
    end = time.time()
    duration = end - start
    analog = duration * 1000000 / 58.0
    # Send  value  Android 
    ser.write(str(analoge).encode('utf-8'))
    
    #### create a SPP on Android device.
    Mac address
    ![image](https://user-images.githubusercontent.com/113662146/226815983-6c3df99b-0bb9-4976-8c11-8dbe872c901a.png)
    rfcomm0 SerialPort for communication


#### Angle xml file link=https://github.com/anvin-shaiju/IntelOneApi-Hackathon/blob/1bff7d1c12a3eb6d769c49d3a0fb1220f21c529e/angle.xml

