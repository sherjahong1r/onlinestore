"""
Created on Mon Nov  3 06:15:00 2025
Daromat va Zararni hisoblaymiz
@author: Sherjahongir
"""
            
            
print("\n\nSizning xarajatlaringizni hisoblaymiz.")

while True:
    savol = input("\nBoshlash uchun start so'zini kiriting"
                  "\nxarajat va zarar uchun son oldiga - qo'ying (-), "
                  "\nTugashi uchun end deb yozing; Namuna: (start/end):"
                  "\n\nNamunani kiriting >>> ").lower()

    if savol == "start":
        try:
            umumiy_foyda = 0
            while True:
                daromat = int(input("\nHisobingizni kiriting: >>> "))
                umumiy_foyda += daromat
                

                again = int(input("Yana son kiritasizmi? (1=Ha / 0=Yo'q): >>> "))
                if again == 0:
                    break

            print(f"\nUmumiy xisob: {umumiy_foyda} ✅")
            if umumiy_foyda > 0:
                print(f"Siz {umumiy_foyda} 000 so'm foyda qilibsiz ")
            if umumiy_foyda == 0:
                print(f"Umumiy xisob {umumiy_foyda} so'm "
                      "foyda ham zarar ham qilmabsiz")
            if umumiy_foyda < 0:
                print(f"siz {umumiy_foyda} 000 so'm zararga kiribsiz")
        except ValueError:
            print("❗ Siz faqat son kiritishingiz mumkin!")
    
    
    elif savol == "end":
        
        print("Hisob tugadi ✅")
        break

    else:
        print("Qaytadan urinib ko‘ring.")

            
            
            


