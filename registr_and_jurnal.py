"""
Created on Sun Nov 14 10:00:36 20
Elektron jurnal
@author: Sherjahongir
"""



print("\nWelkome to elektron jurnal\n")

kirish = input("O'quvchi bo'lsangiz 1 ni \nO'qituvchi bo'lsangiz 2 ni bosing.(1/2)>> ")

if kirish == "1":
   while True:
    ismi = (input("Ismingizni kiriting?>>>").title())
    if ismi:
        print(f"\nHi, student {ismi} "
              "\nTalaba bahosini ko'rish elektron jurnaliga xush kelibsiz!")
        break
    else:
        print('Iltimos ismingizni kiriting!')
    
   while True:
       login = input("\nYangi login kiriting:>>>")
       
       if len(login)<=5: # 5 harfdan ko'proq bo'lishi kerak
           print("Login 5 harfdan ko'proq bo'lishi shart. Qaytadan urining")
       else:
           print("Siz tizimga muvaffaqiyatli kirdingiz!")
           break
    
   talabalar = {
    "Ali": 5,
    "Vali": 4,
    "Davi": 3,
    "Elchin": 5,
    "Shox": 4 }

   malumot = input("\nBahoyingizni ko'rishni hohlaysizmi (1=ha/0=yo'q)>>>")
   while True:
    if malumot == "1":
       print("Siz o'zingizning ismingizni yozishingiz kerak")
       break
    elif malumot == "0":
       print("Thank you for your attention")
       break
    else:
       print("Faqat 1 yoki 0 kiriting!")
       malumot = input("Qayta kiriting (1=ha/0=yo'q)>>> ")
       
   while True:
    ismi = input("\nIsmingizni kiriting>>>").title()
    if ismi in talabalar:
       print(f"{ismi}ning bahosi:", talabalar[ismi])
       break
    else:
       print("Bunday talaba mavjud emas")
       break


elif kirish == "2":
   while True:
     ismi = (input("ismingizni kiriting?>>>").title())
     if ismi:
        print(f"\nHi, teacher {ismi} "
              "\nTalabalarni baholash elektron jurnaliga xush kelibsiz!")
        break
     else:
        print('Iltimos ismingizni kiriting!')
    
   while True:
     login = input("\nYangi login tanlang:>>>")
     
     if len(login)<=5: # 5 harfdan ko'proq bo'lishi kerak
        print("Login 5 harfdan ko'proq bo'lishi shart. Qaytadan urining")
     else:
        print("Siz tizimga muvaffaqiyatli kirdingiz!")
        break
      
   print("\nO'quvchilar ro'yxatini jadvalga kiritamiz.")
   ismlar = []
   n=1 # ismlarni sanash uchun o'zgaruvchi
   while True:
        savol = f"{n}-O'quvchi ismini kiriting>>>"
        ism = input(savol)
        ismlar.append(ism)
        takrorlash = input("Yana ism qo'shasizmi? (1=ha/0=yo'q)")
        n+=1
        if takrorlash != '1':
            break
   print("\nTalabalar ro'yxati:")
   for ism in ismlar:
      royxat = print(ism.title())    
      print(royxat)
      
   royxat = ismlar[:] 
   print("\nTalabalarni baxolaymiz")   
   baholangan_talabalar = {}
   while royxat:
       talaba = royxat.pop() # pop bilan oxiridan boshlab ismlarni oladi.
       while True:
        try:
         baho = input(f"{talaba.title()}ning bahosini kiriting: ")
         print(f"{talaba.title()} baholandi")
         baholangan_talabalar[talaba] = int(baho)
         break  
        except ValueError:
          print("Siz faqat son kiritishingiz kerak")
   
       print("\nUmumiy baholangan talabalar!")
       print(baholangan_talabalar) 
       
       print(baholangan_talabalar.items())
       for key, value in baholangan_talabalar.items():
           print(f"Talaba: {key}")
           print(f"Bahosi: {value} \n")


else:
     print("Faqat 1 yoki 2 ni kiriting!")
     kirish = input("Qayta kiriting (1/2): ")
     

    



