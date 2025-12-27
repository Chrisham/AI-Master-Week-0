age = int(input("Enter your age: "))
has_ticket = False

if age >= 65:
    print("You get a senior discount!")
    if has_ticket:
        print("You can enter")
    else:
        print("You need a ticket to enter")
elif age > 18:
    print("Your ticket price is $15.")
    if has_ticket:
        print("Yu can enter")
    else:
        print("Please purchase a ticket.")
else:
    print("Your tickeet price is 50%")
    if has_ticket:
        print("You can enter")
    else:
        print("Please purchase a ticket.")
    