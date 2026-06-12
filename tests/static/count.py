def count(iterations):
    for i in range(int(iterations)):
        print(i)
    print("done")


if __name__ == "__main__":
    while True:
        user_input = input()
        if "shutdown" in user_input:
            break
        else:
            count(iterations=int(user_input))
