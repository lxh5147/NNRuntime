/*
This file defines the runtime of a neural network.
*/
#ifndef __IMPORT_THEANO_MODEL__
#define __IMPORT_THEANO_MODEL__

#include "nn.hpp"
#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdio>
#include <string>
#include <streambuf>
#include <regex>
#include <vector>

namespace nn_tools {
    using namespace std;
    using namespace nn;
    //Defines NPY binary file model.
    template <typename T>
    class NPYData {
        public:
            //Gets matrix stored in this NPY file.
            shared_ptr<Matrix<T>> getData() {
                if(m_data==nullptr){
                    loadData();
                 }
                 ASSERT(m_data!=nullptr,"m_data");
                 return m_data;
            }
        public:
            NPYData(const string& npyFile):m_file(npyFile){}
        private:
            void loadData(){
                ifstream is(m_file,ios::binary);
                ASSERT(is.is_open(),"is");
                char magic[6];
                read(is,magic,6);
                ASSERT(memcmp("\x93NUMPY", magic, 6)==0,"magic");
                char version[2];
                read(is,version,2);
                int row=1;
                int col=1;
                loadHeaderAndShape(is,row,col);
                T* data=new T[row*col];
                read(is,reinterpret_cast<char*>(data),sizeof(T)*row*col);
                m_data=newMatrix(data,row,col);
            }
            void loadHeaderAndShape(ifstream& is,int& row, int& col){
                unsigned short headerLen;
                char *header = NULL;
                read(is,reinterpret_cast<char*>(&headerLen),sizeof(unsigned short));
                header=new char[headerLen + 1];
                ASSERT(header,"header");
                read(is,header,headerLen);
                header[headerLen] = 0;
                char* shape = strstr(header, "shape");
                ASSERT(shape,"shape");
                shape += strlen("shape");
                shape = strchr(shape, ':');
                ASSERT(shape,"shape");
                ++shape;
                shape = strchr(shape, '(');
                ASSERT(shape,"shape");
                ++shape;
                char* end_shape = strchr(shape, ')');
                ASSERT(end_shape,"end_shape");
                *end_shape = 0;
                col=1;
                auto dim = sscanf(shape, " %d , %d ", &row, &col);
                ASSERT(dim==1||dim==2,"dim");
                delete[] header;
            }
            static void read(ifstream& is,char* buffer, size_t size){
                is.read(buffer,size);
                ASSERT((size_t)is.gcount()==size,"readed");
            }
        private:
            string m_file;
            shared_ptr<Matrix<T>> m_data;
    };

    //Gets the transposed matrix.
    template<typename T>
    shared_ptr<Matrix<T>> transpose(const Matrix<T>& matrix){
        //in-place transpose is complicated when m!=n, refer to https://en.wikipedia.org/wiki/In-place_matrix_transposition.
        size_t row=matrix.col();
        size_t col=matrix.row();
        auto data=new T[row*col];
        ASSERT(data,"data");
        auto source=matrix.data().get();
        for(size_t i=0;i<row;++i){
            for(size_t j=0;j<col;++j){
                data[i*col+j]=source[j*row+i];
            }
        }
        return newMatrix(data,row,col);
    }

    //Defines helper functions to load json file, and to extract values from json string.
    class JsonUtil{
        public:
            //Gets json value for the given key. Enclosed '"' of any json string value are removed.
            static string getJsonValue(const string& jsonContent, const string& key){
                string ex="\"";
                ex+=key;
                ex+="\"";
                ex+=":\\s*";
                //trim '"' when applicable, not greedy match
                ex+="\"?([^,\"\\}]+)";
                regex re(ex);
                smatch match;
                if(regex_search(jsonContent,match,re)){
                    //capture group:0 whole match, 1 first group
                    return match[1];
                }
                else{
                    return "";
                }
            }
            //Gets json array value for the given key. Enclosed '"' of any json string value are removed.
            static vector<string> getJsonStringValues(const string& jsonContent, const string& key){
                string rawValuesExp="\"";
                rawValuesExp+=key;
                rawValuesExp+="\"";
                rawValuesExp+=":\\s*";
                rawValuesExp+="\\[([^\\]]+)\\]";
                regex rawValuesRe(rawValuesExp);
                smatch match;
                vector<string> stringValues;
                if(!regex_search(jsonContent,match,rawValuesRe)){
                    return stringValues;
                }
                //get capture group:0 whole match, 1 first group, example raw value: "prefix_1", "prefix_2"
                //gready match
                string rawStringValues=match[1];
                string valueExp="\"";
                valueExp+="([^\"]+)";
                valueExp+="\"";
                regex valueRe(valueExp);
                while (regex_search (rawStringValues,match,valueRe)) {
                    stringValues.push_back(match[1]);
                    rawStringValues = match.suffix().str();
                }
                return stringValues;
            }
        private:
            JsonUtil(){}
    };

    //Defines theano MLP models, which are saved into a folder.
    template <typename T>
    class TheanoModel{
        public:
            static void load(const string& path, vector<shared_ptr<InputInfo<T>>>& inputsInfo, vector<shared_ptr<Matrix<T>>>& weights, vector<shared_ptr<Vector<T>>>& biasVectors, vector<size_t>& activationFunctionIds, bool includeOutputSoftmaxLayer=false){
                TheanoModel<T> theanoModel(path);
                //load manifest and hyper parameters
                theanoModel.loadManifest();
                theanoModel.loadHyperparams();
                for(const auto& feature:theanoModel.m_sequenceFeatures){
                    theanoModel.loadInputInfo(feature,inputsInfo,true);
                }
                for(const auto& feature:theanoModel.m_nonSequenceFeatures){
                    theanoModel.loadInputInfo(feature,inputsInfo,false);
                }
                //prepare layers
                auto activationFunctionId=getActivationFunctionId(theanoModel.m_activationFunction);
                for(size_t i=0;i<theanoModel.m_numberOfHiddenLayers;++i){
                    string name="h";
                    name+=to_string(i);
                    theanoModel.loadLayer(name,weights,biasVectors);
                    activationFunctionIds.push_back(activationFunctionId);
                }
                if(includeOutputSoftmaxLayer){
                    //append output layer, which is a layer with Identity activation function. For models including the output layer, the model must be loaded with normalizeOutputWithSoftmax=true (which is the default value); for models not including the output layer, the model must be loaded with normalizeOutputWithSoftmax=false.
                    theanoModel.loadLayer("output",weights,biasVectors);
                    activationFunctionIds.push_back(ActivationFunctions<T>::IDENTITY);
                }
            }
        private:
             void loadInputInfo(const string& feature, vector<shared_ptr<InputInfo<T>>>& inputsInfo, bool isSequenceFeature){
                //embedding:W_[feature name].npy, e.g., W_words.npy
                string embeddingFile="W_";
                embeddingFile+=feature;
                embeddingFile+=".npy";
                //one row one embedding
                auto embedding=loadMatrix(getFilePath(m_path,embeddingFile));
                if(isSequenceFeature){
                    inputsInfo.push_back(newInputInfo(EmbeddingWithRawValues<T>::create(*embedding),m_contextLength,m_poolingId));
                }
                else{
                    inputsInfo.push_back(newInputInfo(EmbeddingWithRawValues<T>::create(*embedding)));
                }
            }
            void loadLayer(const string& name, vector<shared_ptr<Matrix<T>>>& weights,vector<shared_ptr<Vector<T>>>& biasVectors){
                //weight:W_[name].npy, e.g., W_h0.npy
                string weightFile="W_";
                weightFile+=name;
                weightFile+=".npy";
                auto weight=loadMatrix(getFilePath(m_path,weightFile));
                //transpose, since theano uses row vector(m*n where m is the number of input nodes) based weight matrix, while runtime uses colum vector based weight matrix
                weights.push_back(transpose(*weight));
                //bias:b_[name].npy, e.g., b_h0.npy
                string vectorFile="b_";
                vectorFile+=name;
                vectorFile+=".npy";
                auto biasVector=loadVector(getFilePath(m_path,vectorFile));
                biasVectors.push_back(biasVector);
            }
            static size_t getActivationFunctionId(const string& activationFunctionName){
                //TODO: support other types of activation functions
                ASSERT(activationFunctionName=="relu"||activationFunctionName=="tanh","activationFunctionName");
                if(activationFunctionName=="relu"){
                    return ActivationFunctions<T>::RELU;
                }
                else{
                    return ActivationFunctions<T>::TANH;
                }
            }
            static shared_ptr<Matrix<T>> loadMatrix(const string& npyFile){
                return NPYData<T>(npyFile).getData();
            }
            static shared_ptr<Vector<T>> loadVector(const string& npyFile){
                auto pMatrix=NPYData<T>(npyFile).getData();
                ASSERT(pMatrix->col()==1,"col");
                size_t size=pMatrix->row();
                auto data = new T[size];
                ASSERT(data,"data");
                memcpy(data,pMatrix->data().get(),sizeof(T)*size);
                return  newVector(data,size);
            }
        private:
            TheanoModel(const string& path):m_path(path), m_poolingId(Poolings<T>::AVG){}
        private:
            void loadManifest(){
                string content=loadTextFileIntoString(getFilePath(m_path,"MANIFEST.json"));
                m_sequenceFeatures=JsonUtil::getJsonStringValues(content,"word_input_features");
                m_nonSequenceFeatures=JsonUtil::getJsonStringValues(content,"query_input_features");
            }
            void loadHyperparams(){
                string content=loadTextFileIntoString(getFilePath(m_path,"hyperparams.json"));
                m_activationFunction=JsonUtil::getJsonValue(content,"activ");
                m_numberOfHiddenLayers=stoi(JsonUtil::getJsonValue(content,"num_hidden"));
                m_contextLength=stoi(JsonUtil::getJsonValue(content,"context"));
            }
            static string getFilePath(const string& root, const string& file){
                string filePath=root;
                filePath+="/";
                filePath+=file;
                return filePath;
            }
            static string loadTextFileIntoString(const string& txtFile){
                //refer to: http://stackoverflow.com/questions/2602013/read-whole-ascii-file-into-c-stdstring
                ifstream is(txtFile);
                ASSERT(is.is_open(),"is");
                is.seekg(0, ios::end);
                string content;
                content.reserve(is.tellg());
                is.seekg(0, ios::beg);
                content.assign((istreambuf_iterator<char>(is)),istreambuf_iterator<char>());
                return content;
            }
        private:
             string m_path;
            //{"query_input_features": ["field_ids"], "word_input_features": ["prefix_1", "prefix_2", "prefix_3", "suffix_1", "suffix_2", "suffix_3", "words"]}
            vector<string> m_sequenceFeatures;
            vector<string> m_nonSequenceFeatures;
            //"num_hidden": 2, "conv": 32, "dropout": 0.5, "batch_size": 100, "feature_dim": 20, "learning_rate_base": 0.0025, "decrease": 0.0, "learning": "n3lu.learning.RMSPROP", "rmsprop_decay": 0.95, "max_epochs": 20, "hidden": 64, "subtensor_adaptive_lr": true, "activ": "relu", "learning_rate_proj": 0.4, "seed": 1234, "patience_epochs": 20, "context": 3,
            string m_activationFunction;
            size_t m_contextLength;
            const int m_poolingId;
            size_t m_numberOfHiddenLayers;
    };

    template<typename T>
    void import_theano_model(const string& inputTheanoModelFolder, const string& outputRuntimeModelFile, bool includeOutputSoftmaxLayer){
        vector<shared_ptr<InputInfo<T>>> inputsInfo;
        vector<shared_ptr<Matrix<T>>> weights;
        vector<shared_ptr<Vector<T>>> biasVectors;
        vector<size_t> activationFunctionIds;
        TheanoModel<T>::load(inputTheanoModelFolder,inputsInfo,weights,biasVectors,activationFunctionIds,includeOutputSoftmaxLayer);
        MLPModelFactory<T>::save(outputRuntimeModelFile,inputsInfo,weights,biasVectors,activationFunctionIds);
    }
}

#endif

