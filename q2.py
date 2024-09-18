def max_rect_area(arr):
    MOD = 10**9 + 7
    # Sort the array in descending order
    arr.sort(reverse=True)
    
    n = len(arr)
    pairs = []
    
    i = 0
    while i < n - 1:
        if arr[i] == arr[i + 1]:  # if both sticks are the same
            pairs.append(arr[i])
            i += 2  # Skip the next one, since we used both sticks
        elif arr[i] - 1 == arr[i + 1]:  # if one stick can be reduced by 1 unit
            pairs.append(arr[i + 1])
            i += 2  # Skip the next one, since we used both sticks
        else:
            i += 1  # Move to the next stick length

    # Now form rectangles by using two pairs at a time
    area_sum = 0
    for j in range(0, len(pairs) - 1, 2):
        area_sum += pairs[j] * pairs[j + 1]
        area_sum %= MOD  # Take mod at each step to avoid overflow
    
    return area_sum

# Example usage
arr = [2, 3, 3, 4, 6, 8, 8, 6]
print(max_rect_area(arr))  # Expected output depends on the exact pairing
