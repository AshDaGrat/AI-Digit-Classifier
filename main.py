import numpy as np
import tkinter as tk
import torch

def main():
    global grid
    grid = np.zeros((28,28))

    model = torch.load('digit_classifier_v4.pt')

    # Create a tkinter window
    window = tk.Tk()
    window.title("AI Digit Classifier")
    window.configure(bg='black')


    # Create a canvas to draw the pixels
    canvas = tk.Canvas(window, width=280, height=280)
    canvas.config(bg='black')

    def get_surrounding_points(x, y):
        surrounding_points = [
            (x-1, y-1), (x, y-1), (x+1, y-1),
            (x-1, y),             (x+1, y),
            (x-1, y+1), (x, y+1), (x+1, y+1)
        ]
        return surrounding_points

    def draw_pixel(event):
        # Calculate the row and column based on the mouse coordinates
        row = event.y // 10
        col = event.x // 10

        # Set the pixel value to black
        grid[row][col] = 1
        for i in get_surrounding_points(row, col):
            if grid[i[0]][i[1]] < 0.7:
                grid[i[0]][i[1]] += 0.3
        
        # Draw the updated pixels
        draw_pixels()


    def draw_pixels():
        # Clear the canvas
        canvas.delete("all")

        # Iterate over the grid and draw the pixels
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == 1:
                    color = "#ffffff" 
                elif grid[i][j] == 0:
                    color = "#000000"
                else:
                    gray_value = int(255 * (grid[i][j]))  # Calculate the gray value based on the input
                    color = '#' + hex(gray_value)[2:].zfill(2) * 3  # Convert gray value to hexadecimal

                canvas.create_rectangle(j * 10, i * 10, (j + 1) * 10, (i + 1) * 10, fill=color)


    def reset():
        global grid
        grid = np.zeros((28,28))
        draw_pixels()


    def convert_grid_to_images_format(grid):
        # Convert `grid` to a NumPy array
        grid_array = np.array(grid)

        # Assuming that the shape of `img` is (1, 784)
        img_shape = (1, 784)

        # Reshape `grid_array` to have the same shape as `img`
        grid_reshaped = grid_array.reshape(*img_shape)

        # Convert `grid_reshaped` to a PyTorch tensor
        img = torch.from_numpy(grid_reshaped).float()

        # Rest of the function code here...
        return(img)


    def find():
        global grid
        img = convert_grid_to_images_format(grid)
        with torch.no_grad():
            logps = model(img)

        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        pred_label = probab.index(max(probab))

        print(pred_label)

    # Bind the mouse events
    canvas.bind("<B1-Motion>", draw_pixel)

    button = tk.Button(window, text="reset", command=reset)
    button.config(bg='black', fg='white')
    button.pack()

    button2 = tk.Button(window, text="find", command=find)
    button2.config(bg='black', fg='white')
    button2.pack()

    # Pack the canvas into the window and start the main loop
    canvas.pack()
    window.mainloop()

if __name__ == '__main__':
    main()