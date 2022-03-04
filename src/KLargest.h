#pragma once
#include <Eigen/Eigen>

//https://www.geeksforgeeks.org/kth-smallestlargest-element-unsorted-array-set-3-worst-case-linear-time/

void swap(double *a, double *b)
{
    double temp = *a;
    *a = *b;
    *b = temp;
}

int partition(double arr[], int l, int r, double x)
{
    // Search for x in arr[l..r] and move it to end
    int i;
    for (i=l; i<r; i++)
        if (abs(arr[i] - x) < 1e-16)
           break;
    swap(arr+i, arr+r);
 
    // Standard partition algorithm
    i = l;
    for (int j = l; j <= r - 1; j++)
    {
        if (arr[j] <= x)
        {
            swap(arr+i, arr+j);
            i++;
        }
    }
    swap(arr+i, arr+r);
    return i;
}

double findMedian(double arr[], int n)
{
    std::vector<double> local_array(arr, arr+n);
    std::sort(local_array.begin(), local_array.end());  // Sort the array
    return local_array[n/2];   // Return middle element
}

// Returns k'th largest element in arr[l..r] in worst case
// linear time. ASSUMPTION: ALL ELEMENTS IN ARR[] ARE DISTINCT
double kthSmallest(double arr[], int l, int r, int k)
{
    // If k is smaller than number of elements in array
    if (k > 0 && k <= r - l + 1)
    {
        int n = r-l+1; // Number of elements in arr[l..r]
 
        // Divide arr[] in groups of size 5, calculate median
        // of every group and store it in median[] array.
        int i;
        double median[(n+4)/5]; // There will be floor((n+4)/5) groups;
        for (i=0; i<n/5; i++)
            median[i] = findMedian(arr+l+i*5, 5);
        if (i*5 < n) //For last group with less than 5 elements
        {
            median[i] = findMedian(arr+l+i*5, n%5);
            i++;
        }   
 
        // Find median of all medians using recursive call.
        // If median[] has only one element, then no need
        // of recursive call
        double medOfMed = (i == 1)? median[i-1]:
                                 kthSmallest(median, 0, i-1, i/2);
 
        // Partition the array around a random element and
        // get position of pivot element in sorted array
        int pos = partition(arr, l, r, medOfMed);
 
        // If position is same as k
        if (pos-l == k-1)
            return arr[pos];
        else if (pos-l > k-1)  // If position is more, recur for left
            return kthSmallest(arr, l, pos-1, k);
        else // Else recur for right subarray
            return kthSmallest(arr, pos+1, r, k-pos+l-1);
    }
 
    // If k is more than number of elements in array
    return -100000000000;
}

double kthLargest(const Eigen::VectorXd& arr, int k)
{
    Eigen::VectorXd neg_arr = -arr;
    return -kthSmallest(neg_arr.data(), 0, neg_arr.size() - 1, k);
}
 