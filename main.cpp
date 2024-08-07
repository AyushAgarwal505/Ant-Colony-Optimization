#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#define N 500     // Number of cities
#define ANTS 500  // Number of ants
#define ALPHA 1.0 // Pheromone importance
#define BETA 2.0  // Heuristic importance
#define RHO 0.5   // Evaporation rate
#define Q 100     // Pheromone deposit constant
#define MAX_ITERATIONS 1000
double distance[N][N];             // Distance matrix between cities
double pheromone[N][N];            // Pheromone matrix
int best_path[N + 1];              // Best path found by ants
double best_path_length = INT_MAX; // Length of the best path
// Calculate distance between two cities
double calculate_distance(int city1, int city2)
{
    // For simplicity, assuming cities are represented in a Euclidean space
    // Distance formula: sqrt((x2 - x1)^2 + (y2 - y1)^2)
    // Here, we assume coordinates are randomly generated between 0 and 1000
    double x1 = rand() % 1000;
    double y1 = rand() % 1000;
    double x2 = rand() % 1000;
    double y2 = rand() % 1000;
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}
// Initialize distance and pheromone matrices
void initialize()
{
    // Initialize distance matrix
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            distance[i][j] = calculate_distance(i, j);
            pheromone[i][j] = 1.0; // Initial pheromone level
        }
    }
}
// Ant's tour construction
void construct_tour(int ant_id, int tour[N + 1])
{
    int visited[N]; // Array to keep track of visited cities
    for (int i = 0; i < N; i++)
    {
        visited[i] = 0;
    }
    // Start from a random city
    int current_city = rand() % N;
    tour[0] = current_city;
    visited[current_city] = 1;
    // Construct tour
    for (int i = 1; i < N; i++)
    {
        double roulette_wheel = 0.0;
        double probabilities[N]
            // Calculate probabilities to move to each city
            for (int j = 0; j < N; j++)
        {
            if (visited[j] == 0)
            {
                probabilities[j] = pow(pheromone[current_city][j], ALPHA) * pow(1.0 / distance[current_city][j], BETA);
                roulette_wheel += probabilities[j];
            }
            else
            {
                probabilities[j] = 0.0;
            }
        }
        // Select next city based on probabilities
        double random_value = ((double)rand() / RAND_MAX) * roulette_wheel;
        double sum = 0.0;
        int next_city = -1;
        for (int j = 0; j < N; j++)
        {
            if (visited[j] == 0)
            {
                sum += probabilities[j];
                if (sum >= random_value)
                {
                    next_city = j;
                    break;
                }
            }
        }
        // Move to the next city
        tour[i] = next_city;
        visited[next_city] = 1;
        current_city = next_city;
    }
    // Return to the starting city to complete the tour
    tour[N] = tour[0];
}
// Update pheromone levels
void update_pheromones()
{
    // Evaporate pheromone
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            pheromone[i][j] *= (1.0 - RHO);
        }
    }
    // Deposit pheromone on best path found by ants
    double delta_pheromone = Q / best_path_length;
    for (int i = 0; i < N; i++)
    {
        int from = best_path[i];
        int to = best_path[i + 1];
        pheromone[from][to] += delta_pheromone;
        pheromone[to][from] += delta_pheromone; // Symmetric matrix
    }
}
// Main function
int main()
{
    srand(time(NULL)); // Seed for random number generation
    initialize();
    clock_t start_time = clock(); // Start time
// Parallelize the ant colony construction process
#pragma omp parallel for
    for (int iter = 0; iter < MAX_ITERATIONS; iter++)
    {
        int local_best_path[N + 1];
        double local_best_path_length = INT_MAX;
        // Each ant constructs a tour
        for (int ant = 0; ant < ANTS; ant++)
        {
            int tour[N + 1];
            construct_tour(ant, tour);
            // Calculate tour length
            double tour_length = 0.0;
            for (int i = 0; i < N; i++)
            {
                tour_length += distance[tour[i]][tour[i + 1]];
            }
            // Update local best path
            if (tour_length < local_best_path_length)
            {
                local_best_path_length = tour_length;
                for (int i = 0; i < N + 1; i++)
                {
                    local_best_path[i] = tour[i];
                }
            }
        }
// Update global best path and pheromones
#pragma omp critical
        {
            if (local_best_path_length < best_path_length)
            {
                best_path_length = local_best_path_length;
                for (int i = 0; i < N + 1; i++)
                {
                    best_path[i] = local_best_path[i];
                }
            }
            update_pheromones();
        }
    }
    clock_t end_time = clock();                                           // End time
    double total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC; // Total time taken
    // Print best path found
    printf("Best path: ");
    for (int i = 0; i < N + 1; i++)
    {
        printf("%d ", best_path[i]);
    }
    printf("\n");
    printf("Best path length: %f\n", best_path_length);
    printf("Total time taken: %f seconds\n", total_time);
    return 0;
}
