package com.example.myapplication;

public interface RecycleVewOnClick {
    void onItemClick(int position);

    void onLongItemClick(int position);

    void onItemDoubleTap(int position);

    void onItemSwipe(int position, int direction);
}