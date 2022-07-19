#pragma once

#include <ML/Vector.h>
#include <ML/Matrix.h>

auto RndRange(double min, double max) -> double;

// Vector
bool operator == (const ml::Vector& a, const ml::Vector& b);

auto operator + (const ml::Vector& a, const ml::Vector& b)->ml::Vector;
auto operator - (const ml::Vector& a, const ml::Vector& b)->ml::Vector;
auto operator * (const ml::Vector& a, const ml::Vector& b)->ml::Vector;
auto operator * (const ml::Vector& a, double b)->ml::Vector;
auto operator * (double a, const ml::Vector& b)->ml::Vector;

auto operator += (ml::Vector& a, const ml::Vector& b)-> const ml::Vector&;
auto operator -= (ml::Vector& a, const ml::Vector& b)-> const ml::Vector&;
auto operator *= (ml::Vector& a, const ml::Vector& b)-> const ml::Vector&;

// Matrix
bool operator == (const ml::Matrix& a, const ml::Matrix& b);

auto operator + (const ml::Matrix& a, const ml::Matrix& b)->ml::Matrix;
auto operator - (const ml::Matrix& a, const ml::Matrix& b)->ml::Matrix;
auto operator * (const ml::Matrix& a, const ml::Matrix& b)->ml::Matrix;
auto operator * (const ml::Matrix& a, double b)->ml::Matrix;
auto operator * (double a, const ml::Matrix& b)->ml::Matrix;

auto operator += (ml::Matrix& a, const ml::Matrix& b)-> const ml::Matrix&;
auto operator -= (ml::Matrix& a, const ml::Matrix& b)-> const ml::Matrix&;

// Vector & Matrix
auto operator * (const ml::Matrix& a, const ml::Vector& b) -> ml::Vector;
auto operator * (const ml::Vector& a, const ml::Matrix& b) -> ml::Vector;

auto operator & (const ml::Vector& a, const ml::Vector& b)->ml::Matrix;
auto operator ^ (const ml::Vector& a, const ml::Vector& b)->double;