//
// Created by root on 7/2/19.
//

#include "MNIST_CPP.h"

tensorflow::Session *MNIST_CPP::loadModel(const string &pbFile, const string &model_name) {
    Status status;
    GraphDef graphDef;
    Session *session = nullptr;
    try {
        status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), pbFile, &graphDef);
        if (!status.ok()) {
            cout << model_name << " Error reading graph file: " << status.ToString().c_str() << endl;
            throw (0);
        }
        session = tensorflow::NewSession(tensorflow::SessionOptions());
        if (session == nullptr) {
            cout << model_name << " Error creating TensorFlow session." << endl;
            throw (1);
        }
        status = session->Create(graphDef);
        if (!status.ok()) {
            cout << model_name << " Error creating graph: " << status.ToString().c_str() << endl;
            throw (2);
        }
        cout << model_name << " init OK." << endl;

    }
    catch (...) {
        cout << model_name << " init failed." << endl;
    }
    return session;
}



MNIST_CPP::MNIST_CPP(string modelpath){
    this->lenetSession =loadModel(modelpath, "Lenet");
}



MNIST_CPP::~MNIST_CPP(){
    if (nullptr != this->lenetSession) {
        tensorflow::Status status = this->lenetSession->Close();
        if (!status.ok())
            TF_CHECK_OK(status);
        else
            cout << "lenet Session Release..." << endl;
    }

    if (lenetSession){
        delete lenetSession;
    }
}


void MNIST_CPP::process(cv::Mat im){

    cv::resize(im,im,cv::Size(input_width,input_height));
    im.convertTo(im, CV_32FC3);
    vector<tensorflow::Tensor> outputTensors;
    tensorflow::Tensor image_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, im.rows, im.cols, 3}));

    im = im/255-0.5;
    float *p = image_tensor.flat<float>().data();
    Mat tempMat(im.rows, im.cols, CV_32FC3, p);
    im.convertTo(tempMat,CV_32FC3);

    tensorflow::Status status = lenetSession->Run({{"input:0", image_tensor}},
                                                 {"prediction:0","probability:0"}, {}, &outputTensors);
    if (!status.ok())
        TF_CHECK_OK(status);

    auto resultEigen_prediction = outputTensors[0].flat<long long>();
    vector<long long> resultVec_prediction(resultEigen_prediction.data(), resultEigen_prediction.data() + resultEigen_prediction.size());
    auto resultEigen_probability = outputTensors[1].flat<float>();
    vector<float> resultVec_probability(resultEigen_probability.data(), resultEigen_probability.data() + resultEigen_probability.size());


    for(int i=0;i<resultVec_prediction.size();i++){
        cout<<resultVec_prediction[i]<<endl;
    }

    for(int i=0;i<resultVec_probability.size();i++){
        cout<<resultVec_probability[i]<<endl;
    }


}
