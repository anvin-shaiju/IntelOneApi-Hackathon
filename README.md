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
#### Use daal4py to improve accuracy and performance of message classification

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
 ```
    Mac address
    rfcomm0 SerialPort for communication
    
    ![image](https://user-images.githubusercontent.com/113662146/226815983-6c3df99b-0bb9-4976-8c11-8dbe872c901a.png)
```
#### connection establishment on android device:

 ![image](https://user-images.githubusercontent.com/113662146/226824961-aaaf6b1a-b509-4631-8598-4c85c9effe86.png)
 //Import The IO package and UUID package to remove error

### Now Intel One API PART and Android Application 

#### Use of math kernal Library.
```
from intel_math import atan2
import math

angle = atan2(sensor_reading, 1000)
angle_deg = math.degrees(angle)
```
### use Of AI analytical Tool.
#### Import  Required libraries.

```
import xml.etree.ElementTree as ET
from openvino.inference_engine import IECore
from daal4py import svd, pca
import numpy as np

tree = ET.parse('angles.xml')
root = tree.getroot()
angle_value = float(root.find('angle').text)
messages = [elem.text for elem in root.findall('messages/message')]
```
#### Determine the appropriate messages for the angle value using OpenVINO.
```
ie = IECore()
model_xml = "angle_model.xml"
model_bin = "angle_model.bin"
net = ie.read_network(model=model_xml, weights=model_bin)
exec_net = ie.load_network(network=net, device_name="CPU")
input_blob = next(iter(net.input_info))
output_blob = next(iter(net.outputs))
angle_tensor = np.array([angle_value], dtype=np.float32)
output = exec_net.infer(inputs={input_blob: angle_tensor})[output_blob]
angle_index = int(np.round(output[0]))
angle_messages = messages[angle_index*4:angle_index*4+4]
pca_model = pca(n_components=3, svd_solver='randomized')
X = np.array([list(map(float, message.split(','))) for message in angle_messages])
X_centered = X - np.mean(X, axis=0)
X_transformed = pca_model.fit_transform(X_centered)
S, V, _ = svd(X_transformed, computeU=False)
selected_message_index = np.argmax(V)
```


#### Now to print the messages in android app.


Compile the C++ code of the OpenVINO and daal4py libraries using the Android NDK.
-Documentation for thee above
https://developer.android.com/ndk/guides/
-Create a JNI wrapper class in your Android app that defines the native methods you want to call from your Java code.

#### C++ Code.

```
InferenceEngine::Core ie;
InferenceEngine::CNNNetReader net_reader;
net_reader.ReadNetwork(xml_file);
net_reader.ReadWeights(bin_file);
InferenceEngine::CNNNetwork network = net_reader.getNetwork();
auto exec_net = ie.LoadNetwork(network, "CPU");
auto input_info = network.getInputsInfo().begin()->second;
auto output_info = network.getOutputsInfo().begin()->second;
InferenceEngine::Blob::Ptr angle_tensor = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, {1}});
angle_tensor->allocate();
float *angle_data = angle_tensor->buffer().as<float *>();
angle_data[0] = angle_value;
InferenceEngine::BlobMap inputs;
inputs[input_info->name()] = angle_tensor;
InferenceEngine::OutputsDataMap outputs_info(network.getOutputsInfo());
auto output_name = outputs_info.begin()->first;
InferenceEngine::BlobMap output;
output[output_name] = InferenceEngine::make_shared_blob
 ```
 
 #### java String Conversion.
 ```
const char *xml_file = env->GetStringUTFChars(xmlFile, nullptr);
const char *bin_file = env->GetStringUTFChars(binFile, nullptr);
const char *messages_xml = env->GetStringUTFChars(messagesXml, nullptr);
 ```
####App final output Screens



## Conclusion

#### In conclusion, a notable advancement in healthcare technology can be seen in the smart belt technology, which combines IntelOneApi IoT and  Intel OneAPi AI analytics to monitor spinal activity, communicate data to an Android app, and deliver therapy through vibrations and impulses The smart belt can offer insights on a person's posture and movements, which might affect their general health and wellness, by tracking spinal activity. The use of IoT technology also makes it possible for data to be collected and transmitted in an efficient manner to an Android app, where it may be remotely examined by medical specialists or the wearer themselves. A useful addition to the smart belt's capabilities is its capacity to deliver treatment through vibrations and impulses. This feature can help users achieve better overall health outcomes by supporting muscle training and rehabilitation.

#### In general, smart belt technology has the ability to enhance people's health and wellbeing by encouraging good behaviors and offering insightful data about how the body works. We may anticipate major gains in patient quality of life and healthcare outcomes as a result of ongoing technological breakthroughs.By boosting the efficiency of AI analytics and expanding the smart belt's processing power, the integration of Intel One API adds a new layer of capability.


## Output Screens

![sample 1 iot](https://user-images.githubusercontent.com/113662146/226892845-5def573f-55d5-4b9c-9310-a3783314b8b5.jpg)
![sample iot 2](https://user-images.githubusercontent.com/113662146/226892817-9cbad78b-10ff-4964-bd87-98f2cc4a6cbf.jpg)



https://user-images.githubusercontent.com/113662146/226891444-2b7953b9-5e77-49cf-91c9-787524cf003f.mp4





