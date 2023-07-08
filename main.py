import numpy as np
import tkinter as tk


def main():
    global grid
    grid = np.zeros((28,28))


    # Create a tkinter window
    window = tk.Tk()
    window.title("AI Digit Classifier")
    window.configure(bg='black')


    # Create a canvas to draw the pixels
    canvas = tk.Canvas(window, width=280, height=280)
    canvas.config(bg='black')


    def draw_pixel(event):
        # Calculate the row and column based on the mouse coordinates
        row = event.y // 10
        col = event.x // 10

        # Set the pixel value to black
        grid[row][col] = 1

        # Draw the updated pixels
        draw_pixels()


    def draw_pixels():
        # Clear the canvas
        canvas.delete("all")

        # Iterate over the grid and draw the pixels
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                color = "white" if grid[i][j] == 1 else "black"
                canvas.create_rectangle(j * 10, i * 10, (j + 1) * 10, (i + 1) * 10, fill=color)


    def reset():
        global grid
        print(grid)
        grid = np.zeros((28,28))
        draw_pixels()


    # Bind the mouse events
    canvas.bind("<B1-Motion>", draw_pixel)

    button = tk.Button(window, text="reset", command=reset)
    button.config(bg='black', fg='white')
    button.pack()

    # Pack the canvas into the window and start the main loop
    canvas.pack()
    window.mainloop()

if __name__ == '__main__':
    main()