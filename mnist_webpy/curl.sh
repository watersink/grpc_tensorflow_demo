base64 ../train_test_mnist/MNIST/testimage/5/1.jpg>image_base64
curl -X POST -F image=@image_base64 'http://localhost:8080/predict'