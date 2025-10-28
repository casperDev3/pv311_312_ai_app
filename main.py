import numpy as np
import time
import matplotlib.pyplot as plt

def main():
    # Create a large random array
    data = np.random.rand(1000000)
    data_two =  ["test", 25, True]

    # Start timing
    start_time = time.time()


    # add 1 to each element in the array
    for i in range(len(data)):
        data[i] += 1

    #  End timing
    end_time = time.time()
    print(f"Time taken to add 1 to each element: {end_time - start_time} seconds")

    # use numpy vectorized operation to add 1 to each element
    start_time = time.time()
    data += 1
    end_time = time.time()
    print(f"Time taken using numpy vectorized operation: {end_time - start_time} seconds")

def draw_plot():
    # Generate some data y = sin(x)
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # Create a plot
    plt.plot(x, y) # визначаємо дані для осі x та y
    plt.title("Sine Wave") # назва графіка
    plt.xlabel("X-axis") # назва осі x
    plt.ylabel("Y-axis") # назва осі y
    plt.grid(True) # додаємо сітку
    plt.show() # відображаємо графік

if __name__ == "__main__":
    # main()
    draw_plot()
