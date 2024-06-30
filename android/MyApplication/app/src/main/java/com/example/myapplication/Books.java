package com.example.myapplication;

public class Books {
    private int img;
    private String name;
    private String author;

    public Books(int img, String name) {
        this.img = img;
        this.name = name;
        this.author = author;
    }

    public int getimg() {
        return img;
    }

    public String getname(){
        return name;
    }

    public String getAuthor() {
        return author;
    }
}