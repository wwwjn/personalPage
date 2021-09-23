---
title: CS144-Lec01-Notes
abbrlink: 18bb33f9
date: 2021-09-22 21:55:00
tags: CS144
---



###  Important Abstraction: Datagrams

"Best-effort Delivery"

Features:

1. has a "TO addres" (a computer's address)
2. has a "FROM address"
3. short text (~1kbyte, <1.5 kbyte)



### Possibilities a datagram could have

```
From address (IPv4 32 bits/ IPv6 128 bits)
To address (v4 32 bits/ v6 128 bits)
Message: "hello"
```

- posibility:
  1. delivered, quickly 
  2. delivered, wrong text (truncate/disordered)
  3. Delivered, really late
  4. Delivered, to wrong address / from wrong address
  5. Never delivered
  6. Delivered, tampered text
  7. Delivered, multiple times
  8. ~~Lives forever~~ (Can not happen, using TTL to limit) 

### Address

- We may Run out of address 2^32(IPv4)

- Calculate 104.196.238.229:
  - 229 + 238\*256 + 196\*256 \*256 + 104 \* 256\* 256\* 256 = 1757736677
  - telnet 104.196.238.229 == telnet 1757736677 [amazing!]
  - address is unique in a network
  - IP address is from network (no from the manufacturer)
  - IP address has distributed, hiearchiral structure(eg, MIT distribute IP address accoring to the computer's building)

#### mtr(traceroute)

Time to Live (TTL) : number of step the datagram allows to have 

- How to traceroute? 
  - using TTL
  - when the TTL = 0, send a message back to the sender

- ==Competition: find a mtr where steps than 35?==



### Byte Stream

Byte Stream: [another abstraction]

- A writer: "abc"
- A reader: "abc", get **same bit to the same order**



Lab0 Task: build a Byte stream from a datagram



### Network Stack/Layers

layers / "network stack" /Internet 4 layes model： 

```
response/request
^
[HTTP] [Application layer] 
^
Byte stream
^
[Transmission Control Protocol (TCP)] [Transport Layer] 
^
datagram [Internet Layer]
^
network interface 
^
wave pocket/frame== [link layer]
```



5 layers model:

```
response/request
^
[TLS(Transport Layer Security)]
^
[HTTP] [Application layer] 
^
Byte stream
^
[Transmission Control Protocol (TCP)] [Transport Layer] 
^
datagram [Internet Layer]
^
network interface 
^
wave pocket/frame [link layer]
```



### how to prentend to be another "from address":

#### Web HTTP proxy

```
rr        proxy         To address
------------------------------------
HTTP   HTTP -> HTTP     HTTP
⬇      ^       ⬇        ^ 
BS      BS      BS       BS  
⬇      ^       ⬇        ^
TCP     TCP     TCP      TCP
⬇      ^       ⬇        ^
DG  ->  DG      DG  ->   DG
```



#### Virtual private network(VPN)

use a different DG (with another "from address")

```
Request/Response
^
[HTTP]
^
BS
^
[TCP]
^
DG1 / DG 2
```



#### TCP proxy (Tor, onion router)

Talk about it later.



### Multiplexing

multiplexing：simply put a signal in each layer to tell which upper layers it should goes to.

eg: The upper layer of IP might be ICMP(a control protocol) or TCP. Using a mark in the Datagram to distinguish
