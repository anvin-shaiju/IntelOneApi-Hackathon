#include <jni.h>
#include <string>
#include <vector>
#include "xml/etree.h"
#include "openvino/inference_engine.hpp"
#include "daal4py/svd.hpp"
#include "daal4py/pca.hpp"

extern "C" JNIEXPORT jobjectArray JNICALL Java_com_example_myapp_AngleDetector_detectAngle(
        JNIEnv *env,
        jclass /* this */,
        jstring xmlFile,
        jstring binFile,
        jstring messagesXml) {

    // Convert Java strings to C++ strings
    const char *xml_file = env->GetStringUTFChars(xmlFile, nullptr);
    const char *bin_file = env->GetStringUTFChars(binFile, nullptr);
    const char *messages_xml = env->GetStringUTFChars(messagesXml, nullptr);

    // Load the angle value and messages from the XML file
    xml::document doc = xml::parse_file(messages_xml);
    double angle_value = std::stod(doc.child("root").child("angle").child_value());
    std::vector<std::string> messages;
    for (auto message : doc.child("root").child("messages").children()) {
        messages.push_back(message.child_value());
    }

    // Determine the appropriate messages for the angle value using OpenVINO and daal4py
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