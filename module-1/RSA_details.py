# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 12:33:13 2021

@author: INBOTICS
"""
import rsa
import binascii
# generate public and private keys with
# rsa.newkeys method,this method accepts
# key length as its parameter
# key length should be atleast 16
publicKey, privateKey = rsa.newkeys(512)
 
# this is the string that we will be encrypting
message = "hello geeks"
 
# rsa.encrypt method is used to encrypt
# string with public key string should be
# encode to byte string before encryption
# with encode method
encMessage = rsa.encrypt(message.encode(),
                         publicKey)
hexilify= binascii.hexlify(encMessage)
 
print("original string: ", message)

print("encrypted string: ", hexilify)
 
print("encrypted string: type ", type(hexilify))

str1 = hexilify.decode('UTF-8')  


stringofdata=str1
print(str1)
convertedtobyte = bytes(stringofdata, 'utf-8')
print(convertedtobyte)
# the encrypted message can be decrypted
# with ras.decrypt method and private key
# decrypt method returns encoded byte string,
# use decode method to convert it to string
# public key cannot be used for decryption
decMessage = rsa.decrypt(binascii.unhexlify(convertedtobyte), privateKey).decode()
 
print("decrypted string: ", decMessage)