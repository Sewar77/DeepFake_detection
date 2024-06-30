package com.example.myapplication;

import static com.example.myapplication.R.drawable;
import static com.example.myapplication.R.raw;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import java.util.ArrayList;

public class MainActivity extends AppCompatActivity implements RecycleVewOnClick {
    @SuppressLint("WrongViewCast")
    public RecyclerView rv;
    private static final int REQUEST_AUDIO_PERMISSION = 200;
    private MediaPlayer mediaPlayer;


    @SuppressLint("WrongViewCast")
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        rv = findViewById(R.id.rc_main);
        ImageView imageView = findViewById(R.id.imagebook);
        checkAudioPermissions();
        imageView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                playAudio();
            }
        });
    }


    public void playAudio() {
        if (mediaPlayer == null) {
            mediaPlayer = MediaPlayer.create(this, raw.audio);
            mediaPlayer.setOnCompletionListener(new MediaPlayer.OnCompletionListener() {
                @Override
                public void onCompletion(MediaPlayer mediaPlayer) {
                    releaseMediaPlayer();
                }
            });
        }
        if (!mediaPlayer.isPlaying()) {
            mediaPlayer.start();
        }


        ArrayList<Books> book = new ArrayList<>();
        book.add(new Books(drawable.cpp, "c++"));
        book.add(new Books(drawable.css, "css"));
        book.add(new Books(drawable.html, "HTML"));
        book.add(new Books(drawable.js, "JAVA SCRIPT"));
        book.add(new Books(drawable.java, "JAVA"));
        book.add(new Books(drawable.python, "PYTHON"));
        book.add(new Books(drawable.php, "PHP"));
        book.add(new Books(drawable.qotlin, "KOTLIN"));
        book.add(new Books(drawable.sql, "SQL SEREVER"));
       /* book.add(new Books(drawable.html, "HTML"));
        book.add(new Books(drawable.js, "JAVA SCRIPT"));
        book.add(new Books(drawable.java, "JAVA"));*/




        RecycleViewAdapter adapter = new RecycleViewAdapter(book, this);
        RecyclerView.LayoutManager lm = new LinearLayoutManager(this);
        rv.setHasFixedSize(true);
        rv.setLayoutManager(lm);
        rv.setAdapter(adapter);

    }

    private void releaseMediaPlayer() {
        if (mediaPlayer != null) {
            mediaPlayer.release();
            mediaPlayer = null;
        }
    }

    @Override
    protected void onStop () {
        super.onStop();
        releaseMediaPlayer();
    }

    private void checkAudioPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.RECORD_AUDIO},
                    REQUEST_AUDIO_PERMISSION);
        } else {
            initializeAudioComponents();
        }
    }

    private void initializeAudioComponents(){}

    @Override
    public void onItemClick(int position) {
        Intent intent = new Intent(String.valueOf(MainActivity.this));
        startActivity(intent);

    }


    @Override
    public void onLongItemClick(int position) {
        // Example: Show a toast message indicating the long click position
        String message = "Long click on item at position: " + position;
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show();
    }

    @Override
    public void onItemDoubleTap(int position) {

    }

    @Override
    public void onItemSwipe(int position, int direction) {

    }



}