# 🗳️ BallotNet

**A secure, transparent, and AI-powered blockchain voting system**

---

## 📖 Overview

BallotNet is a **national e-voting platform** that ensures **security, transparency, and trust** in elections using **blockchain technology** and **AI-based identity verification**.

Voters can securely register and cast their votes through a **Flutter mobile app**, while the **Go backend** handles authentication, blockchain transaction recording, and data integrity.  
The system also integrates **Fayda ID-based KYC** and **facial recognition** for voter verification.

---

## 🚀 Features

- 🔐 **Blockchain-secured voting** — Immutable and transparent ledger for votes  
- 🧠 **AI-based identity verification** — Facial recognition and liveness detection  
- 🪪 **Fayda ID integration** — KYC validation for voter eligibility  
- 📱 **Mobile-first experience** — Built with Flutter for accessibility  
- ⚙️ **Go backend microservices** — Fast, reliable, and scalable server design  
- 📊 **Real-time results** — Transparent vote count and reporting dashboard  

---

## 🏗️ System Architecture

```mermaid
flowchart LR
    A[Voter Mobile App (Flutter)] --> B[API Gateway (Go)]
    B --> C[Identity Service (AI / Python)]
    B --> D[Blockchain Layer (Ethereum / Hyperledger)]
    B --> E[Database (PostgreSQL)]
    D --> F[Audit & Results Dashboard]
