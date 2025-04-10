import matplotlib.pyplot as plt

def plot_final_results(B1plus_map_default, B1plus_map_bestConfig, simulation):

    # Plot the results:  I add a flag to the evaluation function to return the 3D B1+ map
    # Compute B1 map with new config

    B1plus_map_default = B1plus_map_default.detach().numpy()
    B1plus_map_bestConfig = B1plus_map_bestConfig.detach().numpy()

    mask = simulation.simulation_raw_data.subject[:,:,64].detach().numpy()

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(B1plus_map_default[:,:,64]*mask)
    plt.title("B1 - Default fields config")
    plt.colorbar()
    #plt.show()

    plt.subplot(1,2,2)
    plt.imshow(B1plus_map_bestConfig[:,:,64]*mask)
    plt.title("B1 - Best fields config")
    plt.colorbar()
    #plt.show()

    plt.savefig("results.png")

    return 0