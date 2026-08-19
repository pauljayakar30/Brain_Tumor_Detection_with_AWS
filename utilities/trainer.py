import matplotlib.pyplot as plt

from utilities.build_model import build_model
from utilities.data_preparation import prepare, split_data

def train_model(save: bool = True, visualize: bool = True, epochs: int = 30, batch_size: int = 42):
    plt.style.use('dark_background')
    data, result = prepare()
    X_train, X_test, Y_train, Y_test = split_data(data, result)
    model = build_model()
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(X_test, Y_test))

    if save:
        model.save("trained_model", save_format='tf')
    
    if visualize:
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.savefig('loss_graph.png')
        plt.close()

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Test', 'Validation'], loc='lower right')
        plt.savefig('Accuracy_graph.png')
        plt.close()

if __name__ == "__main__":
    train_model(epochs=100)

    
    