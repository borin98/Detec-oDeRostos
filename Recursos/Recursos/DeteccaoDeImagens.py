import cv2

def boundingBox ( imagem, deteccoes ) :
    """
    Função que faz a detecção do objeto de
    interesse e o coloca em um retângulo

    :return:
    """

    for ( x, y, l, a ) in deteccoes:

        cv2.rectangle ( imagem, ( x, y ), ( x + l, y + a ), (0, 255, 0), 2 )

    cv2.imshow("Detector de faces", imagem )
    cv2.waitKey(0)

    return

def main() :

    """
    Função principal que faz a classificação dos dados

    """

    # carregando as imagens
    imagem = cv2.imread("pessoas.jpg")

    # carrega o classificador
    Classificador = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    imagemConvertida = cv2.cvtColor ( imagem, cv2.COLOR_BGR2GRAY )

    # fazendo a classificação dos dados
    deteccoes = Classificador.detectMultiScale ( imagemConvertida,
                                                 scaleFactor=1.09,
                                                 minNeighbors=7,
                                                 minSize=(30, 30),
                                                 maxSize=(70,70) )

    print("Matriz posição de detecção : {}".format(deteccoes))
    print("\nQuantidade de rostos localizados : {}\n".format(len(deteccoes)))

    boundingBox ( imagem, deteccoes )

    cv2.destroyAllWindows()

if __name__ == '__main__':

    """
    Carrega os arquivos iniciais do script
    
    """
    main()