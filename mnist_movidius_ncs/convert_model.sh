mvNCCompile mnist.pb -in=input -on=output -is 28 28 -o mnist.graph
mvNCProfile mnist.pb
mvNCCheck mnist.pb
