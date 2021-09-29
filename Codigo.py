########### Importar Librerías Externas Útiles al Programa ###############

import numpy as np #librería que maneja las matrices
from scipy.optimize import linear_sum_assignment as hungarianAlgorithm #librería donde se define el algoritmo húngaro para resolver el TSP como problema de asignación

#Aclaración: como las listas y matrices en los lenguajes de programación empiezan en un índice 0, ese siempre será el índice inicial.

########### Definición de Clases Útiles ###############

#Clase "Nodo", que define las características y el comportamiento de los nodos en el algorítmo B&B
class Node:
	  def __init__(self, matrix, modified=None, children=[], parent={}): #Para crear un nodo, se toma necesariamente la matriz de adyacencia que plantea como problema de asignación y la arista que debe hacerse inviable (el nodo inicial no tiene este requisito, por defecto no se necesita)
	  																																		#El nodo también puede tomar información sobre sus nodos hijos o su nodo padre
	  	self.matrix = matrix.astype(np.float64)
	  	if(modified!=None):
	  		self.matrix[modified[0],modified[1]] = None #Se hace la modificación pertinente a la versión de la matriz de adyacencia que utiliza el nodo

	  	self.sequence = optimalAssignment(self.matrix) #Se calcula la sequencia del nodo como la asignación óptima para su matriz
	  	self.value = calcWeight(self.matrix, self.sequence) #Se calcula el peso del nodo a partir de la secuencia anterior
	  	self.children = children #Lista con los hijos del Nodo
	  	self.hasChildren = False #Valor booleano que indica si el Nodo tiene hijos o no
	  	self.parent = parent #Padre del Nodo

#Clase "Star", que define las características de una estrella cualquiera que se deba observar
class Star:
  def __init__(self, xRotation, yRotation): #Para crear una estrella, se toman sus coordenadas horizontal (α) y vertical (β)
    self.x = xRotation #Coordenada horizontal (α)
    self.y = yRotation #Coordenada vertical (β)

#Clase "Telescope", que define las características del telescopio en cuestión
class Telescope:
  def __init__(self, xRotation, yRotation): #Para crear un telescopio, se toma el tiempo que le toma, en segundos, rotar un grado horizontalmente (Th) y verticalmente (Ty)
    self.xSpeed = xRotation #Tiempo de rotación horizontal (segundos/grados)
    self.ySpeed = yRotation #Tiempo de rotación vertical (segundos/grados)
    self.maxAngle = 90 * (1+0.5*(yRotation/xRotation)) #Se calcula el ángulo θ, tal que θ = 90*(1+0*5(Ty/Th))
    #Se calcula si el ángulo θ<180, y, por lo tanto, es eficiente usarlo
    if(90 * (1+0.5*(yRotation/xRotation))<180):
    	self.maxAngleEf = True 
    else:
    	self.maxAngleEf = False


########### Definición de Funciones Útiles ###############

#Se define una función para devolver el índice del valor mínimo de cierto índice de una matriz, a lo largo de un cierto índice
def minimumIndex(ogMatrix, axis, indexOnAxis):
	matrix = np.zeros(ogMatrix.shape) + ogMatrix
	
	minimum = np.nan

	if (axis == 0):
		rows = matrix.shape[1]

		for i in range(rows):
			if (str(matrix[indexOnAxis,i])!="nan"): #Se rechaza el valor "nan" como mínimo (a pesar de ser un valor nulo), ya que para este programa "nan" representa un valor infinito (no se puede ingresar "∞" como un valor, en una computadora)
				minimumValue = matrix[indexOnAxis,i]
				minimum = i
				break

		for x in range(rows):
			if ((matrix[indexOnAxis,x] < minimumValue) and (str(matrix[indexOnAxis,i])!="nan")):
				minimumValue = matrix[indexOnAxis,x]
				minimum = x

	elif (axis==1):
		columns = matrix.shape[1]

		for i in range(columns):
			if (str(matrix[i, indexOnAxis])!="nan"):
				minimumValue = matrix[i, indexOnAxis]
				minimum = i
				break

		for x in range(columns):
			if ((matrix[x,indexOnAxis] < minimumValue) and (str(matrix[i, indexOnAxis])!="nan")):
				minimumValue = matrix[x, indexOnAxis]
				minimum = x

	return minimum


#Se define la función que determina una secuencia óptima que resuelva el TSP de cierta matriz según la Heurística del Vecino más Cercano
def closestNeighbour(ogMatrix): #Se toma únicamente la matriz de adyacencia del problema
	matrix = np.zeros(ogMatrix.shape) + ogMatrix #Se crea una matriz idéntica, para no alterar la original
	#Se rechaza la matriz si no es cuadrada
	if not(matrix.shape[0]==matrix.shape[1]):
		return False

	rows = matrix.shape[0] #filas/columnas de la matriz
	ogIndex = 0 #Nodo inicial de la matriz. Se usa el 0 como nodo inicial por un tema de simplicidad
	iIndex = ogIndex #iIndex es el índice del nodo en el que se encuentra la secuencia a cada momento
	sequence = [iIndex] #La secuencia empieza con el nodo inicial = iIndex

	#Se crea una lista con todos los nodos que aún no se han usado en la secuencia
	possibles = []
	for i in range(rows):
		possibles.append(i)
	
	while (len(possibles)>1): #Mientras sigan habiendo nodos sin usar, repetir los siguientes pasos:

		#Ver cuál es, en la matriz de adyacencia, la arista de menor valor para el nodo en el que se encuentra la secuencia
		testIndex = minimumIndex(matrix, 0, iIndex)

		#Si la arista conecta el nodo actual con un nodo aún sin usar en la secuencia, agregar ese siguiente nodo a la sequencia y transformarlo en el nodo actual.
		if (testIndex in possibles): 
			sequence.append(testIndex)
			possibles.remove(iIndex)
			iIndex = testIndex
		
		#Si la arista de menor valor conecta con un nodo ya empleado, transformar el valor de esa arista en "nan" = infinito, para hacerla inviable
		else:
			matrix[iIndex, minimumIndex(matrix, 0, iIndex)]=np.nan

	#Una vez que se hayan usado todos los nodos, agregar a la secuencia el nodo inicial, para cerrarla.
	sequence.append(ogIndex)

	#Devolver la secuencia final.
	return sequence

#Define una función que calcula el peso total de una secuencia cualquiera, para cierta matriz de adyacencia
def calcWeight(matrix, sequenceOfNodes): #Toma la matriz de adyacencia y la secuencia de nodos a usar
	sequence = sequenceOfNodes
	weight = 0 #Define el peso inicial en 0

	#Si la secuencia de nodos es un único recorrido (una única lista), se la inserta dentro de otra lista. Este formateo es necesario porque esta función también calcula el peso de secuencias con múltiples subrrecorridos (que tienen el formato de listas dentro de listas)
	if(type(sequenceOfNodes[0])!=list):
		sequence = [sequenceOfNodes]

	for i in range(len(sequence)):#Por cada subrecorrido de la secuencia

		for x in range(len(sequence[i])-1):#Sea x un nodo en la secuencia
			thisSum = matrix[sequence[i][x], sequence[i][x+1]] #Se toma el valor de la arista que conecta ese nodo y el siguiente nodo en la secuencia de la matriz de adyacencia (que tiene los pesos de cada arista)
			weight = weight + thisSum #Y se añade al peso total

	#Se devuelve el peso total de la secuencia
	return weight

#Se define la función que optimiza el peso de una secuencia según la Heurística de Inversión
def inversionHeur(matrix, ogSequence): #Toma la secuencia a optimizar y la matriz de adyacencia en la que se basa
	sequence = ogSequence

	minimum = calcWeight(matrix, sequence) #Se calcula el peso mínimo a obtener, inicialmente, como el peso de la secuencia a optimizar
	smallSeq = sequence #Se define a la secuencia con menor peso de la misma manera

	#Se calcula la cantidad de nodos (el tamaño) de cada inversión. Se realizan todas las inversiones de todos los tamaños posibles.
	for x in range(2,len(sequence)-2): #El mínimo tamaño de cada inversión es dos. El máximo tamaño es el tamaño de la secuencia a optimizar menos 2.
		sizeInv = x

		for y in range(len(sequence)-1-x): #Se calcula y realiza la máxima cantidad de inversiones posibles en la secuencia de x tamaño. Se resta el 1 para que no se pueda seleccionar para invertir el nodo final.
			start = y+1 #Se calcula el índice en la secuencia del nodo donde comenzará la sección invertida. Se suma el 1 para que no se pueda seleccionar el nodo inicial.
			end = start+sizeInv #Se calcula el índice en el que finaliza la selección a invertir como el índice donde comienza la inversión, más el tamaño de la misma
			newSeq = sequence[start:end] #Se selecciona el subrrecorrido a invertir
			newSeq.reverse() #Se invierte la selección
			#Se añaden a la parte invertida las partes iniciales y finales de la secuencia no alteradas.
			newSeq = sequence[0:start] + newSeq 
			newSeq = newSeq + sequence[end:]

			#Si el peso de la secuencia obtenida es menor al mínimo, se la reemplaza como la secuencia de menor peso.
			if (calcWeight(matrix, newSeq)<minimum):
				minimum = calcWeight(matrix, newSeq)
				smallSeq = newSeq

	#Se devuelve la secuencia de menor peso
	return smallSeq


#Se define la función que devuelva la secuencia resultado de resolver el problema de asignación asociado a una matriz de adyacencia. Se la devuelve en el formato [peso, sequencia]. 
#Es necesario definir esta función ya que la librería importada previamente devuelve el resultado del problema de asignación en un formato inconveniente para este programa; por lo que se lo debe reformatear.
def optimalAssignment(ogMatrix): #Se toma la matriz de adyacencia
	matrix = np.zeros(ogMatrix.shape) + ogMatrix #Se crea una copia de la misma, para no alterar la original.
	points = [] #Se crea una lista donde irán todos los coeficientes óptimos (resultados del método húngaro), según sus índices (i,j) en la matriz de adyacencia.
	sequences = [] #Se crea una lista donde irán todos los nodos de la secuencia de forma ordenada

	#Primero, es necesario reemplazar los valores "nan" = infinito de nuestra matriz de adyacencia, puesto que la función que realiza el método húngaro no los reconoce como válidos.
	#Se los reemplaza con un valor 10 unidades mayor al valor anteriormente mayor (en la matriz original). Esto los hace inviables como coeficientes óptimos.
	for i in range(matrix.shape[0]):
		for j in range(matrix.shape[1]):
			if(str(matrix[i, j])=="nan"):
				matrix[i,j]=0
	for i in range(matrix.shape[0]):
		for j in range(matrix.shape[1]):
			if(matrix[i, j]==0):
				matrix[i,j] = np.amax(matrix)+10

	#Se realiza el método húngaro en la matriz, y se asigna a la variable col_ind los índices en los que están los coeficientes óptimos en cada columna de la matriz.
	row_ind, col_ind = hungarianAlgorithm(matrix)

	#Se agrega los coficientes a la lista points, formateados en sus índices (i,j)
	for x in range(len(col_ind)):
		points.append([row_ind[x],col_ind[x]])

	
	iteration = 0 #Esta es una variable de itineración, que sirve para indicar el subrrecorrido que se está creando. De haber un solo recorrido, sin subrrecorridos, la variable indicará siempre 0.
	numPoints = len(points) #Refiere al número de coeficientes iniciales
	usedPoints=[] #Lista que contiene todos los coeficientes (aristas) ya usados 

	while(True):
		sequences.append([])
		starting, nextNode = points[0] #Se define el primer índice del coeficiente inicial como el nodo inicial, y su siguiente índice como el siguiente nodo.
		usedPoints.append(points[0]) #Se agrega el coeficiente inicial a la lista de usados.
		(sequences[iteration]).append(starting) #Se agrega al subrecorrido inicial el nodo inicial

		conditionWhile = True #Variable que define si se continua la siguiente serie de pasos.
		while(conditionWhile):
			for y in points: #Por cada coeficiente sin usar:
				
				if (y[0]==nextNode):#Si el primer índice del coeficiente corresponde con el nodo que sigue en la secuencia:
					sequences[iteration].append(nextNode) #Agregar el nodo que sigue a la secuencia.
					nextNode = y[1] #Hacer el segundo índice del coeficiente el siguiente nodo.
					usedPoints.append(y) #Agregar este coeficiente a la lista de usados

					if(nextNode==starting): #Si el nodo que sigue es el inicial del subrrecorrido, cerrar el subrrecorrido y:
						sequences[iteration].append(nextNode) #Agregar ese nodo al subrrecorrido

						if((len(usedPoints))==numPoints): #Si todos los coeficientes han sido usados en la secuencia:

							if(len(sequences)==1): #Si la secuencia es un solo subrecorrido, sacarlo de la lista externa. Esto deja al recorrido en una sola lista.
								sequences = sequences[0] 

							return sequences #Finalmente, devolver la secuencia

						else: #Sino, simplemente cerrar el subrecorrido y comenzar el nuevo subrrecorrido
							iteration += 1
							conditionWhile=False
							break

		if(len(usedPoints)!=0): #Si hay ya usados, remover de la lista de puntos posibles los puntos usados.
			for x in usedPoints:
				if(x in points):
					points.remove(x)

#Se define una función para crear hijos de los nodos en el algoritmo B&B. No se crea esta función como método, porque los nodos hijos creados por ese metodo no tendrían la función misma definida (no hay recurción en el código).
def createChildren(node): #Toma un nodo

	if not(node.hasChildren): #Si el nodo no tiene hijos:

		#Si la secuencia del nodo es única, no se pueden crear hijos. Esto es porque los hijos se crean a partir de un subrrecorrido en la secuencia del nodo. 
		#Si la secuencia es un único recorrido, no hay por qué o de dónde crear hijos.
		if(type(node.sequence[0]) is not list): 
			return False
		elif ((type(node.sequence[0]) is list) and (len(node.sequence)==1)):
			return False

		else: #Si la secuencia del nodo, por el contrario, tiene subrecorridos:

			#Se definen al índice del subrrecorrido más corto y su tamaño. Por defecto, se toma el primer subrrecorrido.
			shortestInd = 0
			shortestLen = len(node.sequence[0])

			#Luego se chequea cada subrrecorrido en la secuencia del nodo para identificar verdaderamente el subrrecorrido más corto.
			for x in node.sequence:
				if (len(x)<shortestLen):
					shortestLen = len(x)
					shortestInd = node.sequence.index(x)
			shortest = node.sequence[shortestInd]

			#Para cada arista distinta del subrrecorrido más corto, crear un nodo hijo por hacer esa arista inviable. Poner todos esos nodos hijos en la lista de hijos del nodo inicial.
			for y in range(len(shortest)-1):
				node.children.append(Node(node.matrix, modified=[shortest[y],shortest[y+1]]))

			#A cada nodo hijo, asignarle el nodo inicial como su padre.
			for x in node.children:
				x.parent = node
			node.hasChildren = True #Afirmar que el nodo inicial tiene hijos.

			#Ordenar la lista de nodos hijos por orden creciente del peso de su secuencia.
			node.children.sort(key=lambda x: x.value)

			return node.children #Devolver la lista de nodos hijos
	
	else: #Si, por el contrario, el nodo tiene hijos, no se le pueden volver a crear más hijos.
		return False


#definir la función que realiza el algorítmo de B&B (Ramificación y Acotación)
def branchAndBound(ogMatrix, ogUpperBoundSeq): #Toma la matriz de adyacencia sobre la cual trabaja y la secuencia de la cota mayor
	matrix = np.zeros(ogMatrix.shape) + ogMatrix #Se crea una copia de la matriz de adyacencia para no modificar la original

	#Se crean variables a partir de la secuencia de la cota mayor y su peso
	upperBoundSeq = ogUpperBoundSeq
	upperBound = calcWeight(matrix, upperBoundSeq)
	
	#Se deine el nodo inicial, el primer nodo padre, usando la matriz de adyacencia original
	parentNode = Node(matrix)
	actualNode = parentNode


	while(True): #Se repite la siguiente serie de pasos, hasta que se encuentre una secuencia óptima del algoritmo
		
		if(actualNode.value <= upperBound): #Si el peso del nodo es menor o igual al de la cota mayor:

			if (not(actualNode.hasChildren)): #Si el nodo no tiene hijos:

				if(type(actualNode.sequence[0]) is not list): #Si la secuencia del nodo es un único recorrido

					#Se define a la secuencia del nodo actual como la cota mayor
					upperBoundSeq = actualNode.sequence
					upperBound = actualNode.value

					#Se devuelve una lista con el peso y la secuencia de este nodo, de esta cota mayor. 
					#El algoritmo devuelve la primera secuencia que se encuentre con las condiciones que cumplan con las condiciones para ser respuesta al TSP.
					#Por lo tanto, no realiza un sondeo a fondo.
					return [upperBound, upperBoundSeq]

				else: #Si la secuencia del nodo es un conjunto de subrrecorridos. Crear sus nodos hijos.
					hijos = createChildren(actualNode)
					for x in hijos: #Se aclara que los nodos hijos recién creados aún no deberían tener hijos propios
						x.hasChildren = False
						x.children = []
					actualNode = hijos[0] #Se continúa el algoritmo analizando el nodo hijo de menor peso (el primero, ya que la lista está ordenada según el peso cresciente)

			else: #Si el nodo ya tiene hijos (y, por lo tanto implica que ya ha sido analizado y su secuencia es un subrrecorrido):

				if(actualNode.children != []): #Si aún le quedan hijos sin evaluar:
					
					#Seleccionar el siguiente hijo como el siguiente nodo a analizar
					hijo = actualNode.children[0]
					actualNode = hijo

				else: #Si al nodo ya no le quedan hijos por analizar

					#Eliminar el camino marcado por este nodo como un camino que todavía puede dar una solución óptima
					delNode = actualNode
					#Determinar al padre del nodo actual como el siguiente nodo a analizar
					actualNode = actualNode.parent
					#Eliminar al nodo anterior de la lista de nodos hijos del padre
					actualNode.children.remove(delNode)

		else: #Si el peso del nodo es mayor al de la cota mayor:

			#Eliminar el camino marcado por este nodo como un camino que todavía puede dar una solución óptima
			delNode = actualNode
			actualNode = actualNode.parent
			actualNode.children.remove(delNode)

########### Código a Ejecutar Activamente (y Código Visible al Usuario) ###############

print("*** Todos los ángulos deben ser tratados como ángulos sexagesimales ***") #Aclaración sobre los datos a ingresar


#Definir las características del telescopio a usar
tlscope = Telescope(float(input("Ingresar la velocidad de rotación horizontal del telescopio (segundos/grados): ")), float(input("Ingresar la velocidad de rotación vertical del telescopio (segundos/grados): ")))

rawData = [] #Definir lista donde irán las estrellas en primera instancia
noneMatrix = np.array((None), dtype=float) #Definir una matriz cuyo único valor sea "nan" = infinitp
more = True #Aclarar que siguen habiendo entradas a realizar por el usuario

while (more): #Mientras sigan habiendo entradas a realizar por el usuario
	#Entrar las coordenadas de ángulo horizontales de la estrella
	inputX = float(input("Ingresar las coordenadas de ángulo horizontal de la región celeste (grados): "))
	#Entrar las coordenadas de ángulo verticales de la estrella
	inputY = float(input("Ingresar las coordenadas de ángulo vertical de la región celeste (grados): "))
	#Agregar a la lista de estrellas un nuevo objeto "Star" con las coordenadas previamente ingresadas
	rawData.append(Star(inputX, inputY))
	#Preguntar al usuario si tiene más entradas
	goOn = str(input("¿Hay más entradas?(S/N) "))
	#Si el usuario no tiene más entradas, definir que no se realizaran más entradas
	if (goOn=="N"):
		more = False

#Crear la estructura de la matriz de adyacencia del grafo definido por las estrellas previamente ingresadas
starAdjMatrix = np.ones((len(rawData),len(rawData)))


for x in range(len(rawData)): #Por cada nodo en el grafo de adyacencia (x es el nodo)
	starAdjMatrix[x, x] = None #Definir las diagonales de la matriz como "nan" = infinito

	if not(x+1==len(rawData)): #Si el índice siguiente en la secuencia se encuentra en la matriz de adyacencia. Es decir, si existe un nodo siguiente, de índice mayor al actual a considerar.
		for y in range(x+1,len(rawData)): #Por cada nodo en la secuencia distinto al actual cuya arista que los conecta todavía no ha sido pesada y agregada a la matriz de adyacencia. (y es el siguiente nodo)
			xDiff = abs(rawData[x].x - rawData[y].x) #el módulo de la diferencia entre los ángulos horizontales de los nodos que se están teniendo en cuenta
			yDiff = abs(rawData[x].y - rawData[y].y) #el módulo de la diferencia entre los ángulos verticales de los nodos que se están teniendo en cuenta
			weightXY = 0 #Se asigna el peso a una variable

			#Se define el peso de la arista que conecta los nodos x e y usando las fórmulas y restricciones aclaradas en el desarrollo de la exploración
			if (tlscope.maxAngleEf and ((xDiff<=180 and xDiff > tlscope.maxAngle) or (xDiff>180 and (360-xDiff)>tlscope.maxAngle))):
				weightXY = abs(180-xDiff) * tlscope.xSpeed + (90 + yDiff) * tlscope.ySpeed
			else:
				if(xDiff<=180):
					weightXY = xDiff * tlscope.xSpeed + yDiff * tlscope.ySpeed
				else:
					weightXY = (360 - xDiff) * tlscope.xSpeed + yDiff * tlscope.ySpeed

			#Se agrega en los lugares correspondientes, en los coeficientes (x,y) y (y,x), el peso de la arista.
			starAdjMatrix[x,y] = starAdjMatrix[y, x] = weightXY

#Se muestra al usuario la matriz de adyacencia del grafo del problema que ingresó
print("Matriz de Adjacencia del Grafo de las Estrellas:")
print(starAdjMatrix)
print(" ")

#Se calcula la secuencia óptima de la matriz usando la heurística del vecino más cercano y se muestra al usuario, con su peso
vecinoCercano = closestNeighbour(starAdjMatrix)
vecinoCercanoPeso = calcWeight(starAdjMatrix, vecinoCercano)
print("La sequencia obtenida con la heurística del Vecino más Cercano es: "+str(vecinoCercano)+", con un peso de " + str(vecinoCercanoPeso)+" segundos.")
print(" ")

#Se calcula la secuencia óptimizada del resultado del vecino más cercano con la heurística de inversión y se muestra al usuario, con su peso
inversion = inversionHeur(starAdjMatrix, vecinoCercano)
inversionPeso = calcWeight(starAdjMatrix, inversion)
print("La sequencia obtenida con la heurística de Inversión, a partir de la anterior, es: " + str(inversion) + ", con un peso de " +str(inversionPeso) + " segundos.")
print(" ")

#Se pregunta al usuario si quiere continuar con el cálculo del algoritmo B&B. En caso afirmativo, se calcula la secuencia óptima usando el algoritmo, y se la muestra, con su peso, al usuario
continuar = str(input("Para una cantidad de regiones celestes a observar relativamente pequeña, las aproximaciones anteriores dan un resultado bastante óptimo. La búsqueda de la sequencia más óptima usando un algoritmo exacto tardará más tiempo. ¿Desea aún así continuar con un algorítmo exacto? (S/N) "))
if(continuar == "S" or continuar == "s"):
	acotacion = branchAndBound(starAdjMatrix, inversion)
	print("La sequencia óptima encontrada usando el algoritmo de Ramificación y Acotamiento, usando la sequencia anterior como cota mayor, es:" + str(acotacion[1]) + ", con un peso de " + str(acotacion[0]) + " segundos.")

	
"""
REFERENCIAS:
-GeeksforGeeks. (4 de marzo de 2020). 'Hungarian algorithm for ASSIGNMENT Problem: Set 1 (Introduction)'. GeeksforGeeks. Recuperado el 3 de julio de 2021, de https://www.geeksforgeeks.org/hungarian-algorithm-assignment-problem-set-1-introduction/. 
-The Scipy community. (2016). 'Scipy Documentation: optimize.linear_sum_assignment'. SciPy v0.18.1 Reference Guide. Recuperado el 3 de julio de 2021, de https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html. 
-The NumPy community. (2021). 'Overview - NumPy V1.21 Manual: Documentation'. NumPy v1.21 Manual. Recuperado el 3 de julio de 2021, de https://numpy.org/doc/stable/. 
-Hackebeil, G. A. (2019). 'Welcome to pybnb - 0.6.2 - Documentation'. PyBnB - 0.6.2. Recuperado el 3 de julio de 2021, de https://pybnb.readthedocs.io/en/stable/index.html. 
-GeeksforGeeks. (12 de junio de 2020). 'Traveling salesman problem using branch and bound'. GeeksforGeeks. Recuperado el 3 de julio de 2021, de https://www.geeksforgeeks.org/traveling-salesman-problem-using-branch-and-bound-2/. 
"""
