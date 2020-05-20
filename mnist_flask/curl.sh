imageName="../train_test_mnist/MNIST/testimage/5/1.jpg"
md=`md5sum $imageName`
curl -X POST 'http://localhost:5000/predict' \
     -F "image=@$imageName" \
     -F "md5=`echo $md|awk '{print$1}'`" 
