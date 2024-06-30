package com.example.myapplication;

import android.annotation.SuppressLint;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import java.util.ArrayList;

public class RecycleViewAdapter extends RecyclerView.Adapter<RecycleViewAdapter.BookViewHolder> {
    ArrayList<Books> book;
    private RecycleVewOnClick recycleVewOnClick;


    public RecycleViewAdapter(ArrayList<Books> book, RecycleVewOnClick recycleVewOnClick) {
        this.book = new ArrayList<>(book);
        this.recycleVewOnClick = recycleVewOnClick;
    }


    @NonNull
    @Override
    public BookViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View v = LayoutInflater.from(parent.getContext()).inflate(R.layout.book_item, parent, false);
        return new BookViewHolder(v);
    }

    @SuppressLint("NotifyDataSetChanged")
    public void updateData(ArrayList<Books> newBooks) {
        book.clear();
        book.addAll(newBooks);
        notifyDataSetChanged();
    }

    @Override
    public void onBindViewHolder(@NonNull BookViewHolder holder, int position) {
        Books b = book.get(position);
        if (b != null) {
            holder.img_vw.setImageResource(b.getimg());
            holder.txt_vw.setText(b.getname());
        }
    }


    @Override
    public int getItemCount() {
        return book.size();
    }

    class BookViewHolder extends RecyclerView.ViewHolder {
        TextView txt_vw;
        ImageView img_vw;
        public BookViewHolder(@NonNull View itemview) {
            super(itemview);
            txt_vw = itemview.findViewById(R.id.textbook1);
            img_vw = itemview.findViewById((R.id.imagebook));
            img_vw.setOnClickListener(v -> recycleVewOnClick.onItemClick(getAdapterPosition()));
            itemview.setOnLongClickListener(new View.OnLongClickListener() {
                @Override
                public boolean onLongClick(View v) {
                    recycleVewOnClick.onLongItemClick(getAdapterPosition());
                    return true;
                }
            });
        }

    }
}
