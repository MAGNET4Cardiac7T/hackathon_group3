import matplotlib.pyplot as plt

def plot_final_results(B1plus_map_default, B1plus_map_bestConfig, SAR_default, SAR_bestConfig, simulation):

    # Plot the results:  I add a flag to the evaluation function to return the 3D B1+ map
    # Compute B1 map with new config

    B1plus_map_default = B1plus_map_default.detach().numpy()
    B1plus_map_bestConfig = B1plus_map_bestConfig.detach().numpy()

    SAR_default = SAR_default.detach().numpy()
    SAR_bestConfig = SAR_bestConfig.detach().numpy()

    mask = simulation.simulation_raw_data.subject[:,:,64].detach().numpy()

    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(B1plus_map_default[:,:,64]*mask)
    plt.title("B1 - Default fields config")
    plt.colorbar()
    #plt.show()

    plt.subplot(2,2,2)
    plt.imshow(B1plus_map_bestConfig[:,:,64]*mask)
    plt.title("B1 - Best fields config")
    plt.colorbar()


    plt.subplot(2,2,3)
    plt.imshow(SAR_default[:,:,64]*mask)
    plt.title("SAR - Default fields config")
    plt.colorbar()
    #plt.show()

    plt.subplot(2,2,4)
    plt.imshow(SAR_bestConfig[:,:,64]*mask)
    plt.title("SAR - Best fields config")
    plt.colorbar()
    #plt.show()

    plt.tight_layout()

    plt.savefig("results.png")

    return 0