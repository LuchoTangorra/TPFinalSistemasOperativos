# Trabajo final sistemas operativos
Ingenieria de sistemas, UNICEN.

### Introduccion  
Este informe presenta un análisis acerca del uso de recursos en diferentes tamaños de batch y diferentes tipos de ejecución (imperativa y mediante grafos) sobre modelos de deep learning. Para la implementación de los modelos se utilizó TensorFlow 2.1, junto con su módulo keras, el cual facilita el prototipado.
En este trabajo se realizó una comparación sobre los mismos teniendo en cuenta el tiempo de ejecución, consumo de memoria del GPU, consumo de memoria RAM y pérdida (loss) con la que las redes neuronales clasifican los casos de prueba (test).
A su vez, se implementaron dos tipos de redes neuronales: CNN (Convolutional Neural Network) y RNN (Recurrent Neural Network) con el objetivo de probar las diferencias presentes en ambas cuando se aplican los cambios planteados anteriormente.

### Conclusiones
Puede verse que el modo de ejecución imperativa es más lento en ejecución y, aunque no fue demostrado en este trabajo, más rápido para prototipar. Gracias a la funcionalidad añadida en tensorflow 2.x, podremos beneficiarnos de ambos modos de ejecución, implementando y testeando el modelo de forma imperativa y, luego, modificar ligeramente el código para indicar que se desea utilizar la ejecución por grafo.
Además de beneficiarse del modo de ejecución, podremos ver beneficiado el modelo en precisión y reducción del valor de pérdida eligiendo correctamente el tamaño de batch. Aunque si bien no hay una valor correcto para cada dataset, siempre se recomienda utilizar tamaños potencia de 2 y, según el hardware que tengamos disponible, podremos elegir batches de tamaño 16, 32, 64 o 128. No recomendaría utilizar valores más bajos ni más altos ya que los primeros son altamente ineficientes y los segundos favorecen el overfitting del modelo. Con el tamaño correcto de batch podremos obtener el mayor beneficio de nuestra unidad de procesamiento gráfica y el suficiente ruido en cada batch, evitando de este modo el overfitting. 
Nos podremos dar cuenta si el tamaño de batch que elegimos es el correcto comparando la perdida de entrenamiento con la pérdida en la validación, como se explicó anteriormente en el apartado de Batch. Hay que tener en cuenta que puede darse el caso de que un tamaño que favorezca la performance, perjudique la precisión del modelo, aunque todo ésto dependerá del modelo implementado y de los datos que poseamos. 


En el **Informe final SO.pdf** se encuentran detallados todos los descubriemientos realizados.
