forest_map = input("Enter the forest map (2D list format): ")
center = input("Enter the center coordinates (tuple format): ")
m = int(input("Enter the size of the square (odd integer): "))
forest_map = eval(forest_map)
center = eval(center) 


# forest_map = [
#     [0, 0, 1, 0, 0],
#     [0, 1, 1, 1, 1],
#     [0, 1, 1, 0, 0],
#     [1, 0, 1, 1, 0],
#     [0, 1, 0, 0, 0]
# ]
# center = (2, 2)
# m = 3


print(forest_map)
print(center)
print(m)

def count_trees():
    count = 0
    if forest_map[center[0]][center[1]] != 1:
        print("The center is not a lal chandan")
    else:
        for i in range(center[0] - m//2, center[0] + m//2 + 1):
            for j in range(center[1] - m//2, center[1] + m//2 + 1):
                if forest_map[i][j] == 1:
                    count += 1
        print(count)

count_trees()