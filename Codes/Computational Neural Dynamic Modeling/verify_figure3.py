
import sys
import os
sys.path.append(r'd:\Data\OneDrive\Papers\SWIM\Codes\Computational Neural Dynamic Modeling')
import plot_neural_dynamics_figures as pnd

def main():
    print("Loading data...")
    try:
        data = pnd.load_all_data()
        print("Data loaded. Plotting Figure 3...")
        pnd.plot_figure3(data)
        print("Figure 3 generated successfully.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
