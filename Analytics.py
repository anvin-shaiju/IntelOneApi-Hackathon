import xml.etree.ElementTree as ET
from openvino.inference_engine import IECore
from daal4py import svd, pca
import numpy as np

# Load the angle value and messages from the XML file
tree = ET.parse('messages.xml')
root = tree.getroot()
angle_value = float(root.find('angle').text)
messages = [elem.text for elem in root.findall('messages/message')]

# Determine the appropriate messages for the angle value using OpenVINO and daal4py
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

# Select the four messages for the given angle index
angle_messages = messages[angle_index*4:angle_index*4+4]

# Use daal4py to improve accuracy and performance of message classification
pca_model = pca(n_components=3, svd_solver='randomized')
X = np.array([list(map(float, message.split(','))) for message in angle_messages])
X_centered = X - np.mean(X, axis=0)
X_transformed = pca_model.fit_transform(X_centered)
S, V, _ = svd(X_transformed, computeU=False)
selected_message_index = np.argmax(V)

# Print all four messages
print("Messages for angle index", angle_index)
for i, message in enumerate(angle_messages):
    print("Message", i+1, ":", message)
    
# Display the selected message
selected_message = angle_messages[selected_message_index]
print("Selected message:", selected_message)
