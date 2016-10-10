"""
parallel program
To run in your computer write in the terminal
mpiexec -n 4 python [file_name.py]
where 4 is the arbitrary number of processors.
Must install mpi4py using (pip install mpi4py) in terminal
"""
import numpy as np
from matplotlib import pyplot as plt
import random
#random.seed(1) # so results don't change every time I execute
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

wt = MPI.Wtime()

k1 = 0.015
k2 = 0.00004
k3 = 0.0004
k4 = 0.04
end_time = 600

def get_rates(rabbits, foxes):
    """
    Return the rates (expected events per day) as a tuple:
    (rabbit_birth, rabbit_death, fox_birth, fox_death)
    """
    rabbit_birth = k1 * rabbits 
    rabbit_death = k2 * rabbits * foxes
    fox_birth = k3 * rabbits * foxes 
    fox_death = k4 * foxes
    return (rabbit_birth, rabbit_death, fox_birth, fox_death)

dead_foxes = 0
dead_everything = 0
runs = int(10000/size)

second_peak_times = []
second_peak_foxes = []

mean_times = np.zeros(runs)
mean_foxes = np.zeros(runs)
upper_quartile_times = np.zeros(runs)
lower_quartile_times = np.zeros(runs)
upper_quartile_foxes = np.zeros(runs)
lower_quartile_foxes = np.zeros(runs)


for run in range(runs):
    time = 0
    rabbit = 400
    fox = 200
    # we don't know how long these will be so start as lists and convert to arrays later
    times = []
    rabbits = []
    foxes = []

    while time < end_time:
        times.append(time)
        rabbits.append(rabbit)
        foxes.append(fox)
        (rabbit_birth, rabbit_death, fox_birth, fox_death) = rates = get_rates(rabbit, fox)
        sum_rates = sum(rates)
        if sum_rates == 0:
            # print("everything dead at t=",time)
            dead_everything += 1
            times.append(end_time)
            rabbits.append(rabbit)
            foxes.append(fox)
            break
        wait_time = random.expovariate( sum_rates )
        time += wait_time
        choice = random.uniform(0, sum_rates)
        # Imagine we threw a dart at a number line with span (0, sum_rates) and it hit at "choice"
        # Foxes change more often than rabbits, so we'll be faster if we check them first!
        choice -= fox_birth
        if choice < 0:
            fox += 1 # fox born
            continue
        choice -= fox_death
        if choice < 0:
            fox -= 1 # fox died
            if fox == 0:
                #print("Foxes all died at t=",time)
                dead_foxes += 1
                ## Break here to speed things up (and not track the growing rabbit population)
            continue
        if choice < rabbit_birth:
            rabbit += 1 # rabbit born
            continue
        rabbit -= 1 # rabbit died
    
    times = np.array(times)
    rabbits = np.array(rabbits)
    foxes = np.array(foxes)
    
    index_of_second_peak = np.argmax(foxes*(times>200)*(foxes>100))
    if index_of_second_peak:
        second_peak_times.append(times[index_of_second_peak])
        second_peak_foxes.append(foxes[index_of_second_peak])
    
    if len(second_peak_times)>0:
        mean_times[run] = np.mean(second_peak_times)
        mean_foxes[run] = np.mean(second_peak_foxes)
        upper_quartile_times[run] = np.percentile(second_peak_times,75)
        lower_quartile_times[run] = np.percentile(second_peak_times,25)
        upper_quartile_foxes[run] = np.percentile(second_peak_foxes,75)
        lower_quartile_foxes[run] = np.percentile(second_peak_foxes,25)

    # We don't want to plot too many lines, but would be fun to see a few
    if run < 20 :
        plt.plot(times, rabbits, 'b')
        plt.plot(times, foxes, 'g')

comm.Barrier()
if rank == 0:
	wt = MPI.Wtime() - wt
	print ("run time {:.2f} with number of runs {} and number of processors {}".format(wt, runs*size, size))
comm.Barrier()

plt.legend(['rabbits','foxes'],loc="best") # put the legend at the best location to avoid overlapping things
plt.ylim(0,3000)
plt.title("This plot from processor {}".format(rank))
plt.show()

total_mean_times = np.zeros(1)
comm.Reduce(mean_times[-1] , total_mean_times , op=MPI.SUM , root=0)
total_mean_times = total_mean_times /size

total_mean_foxes = np.zeros(1)
comm.Reduce(mean_foxes[-1] , total_mean_foxes , op=MPI.SUM , root=0)
total_mean_foxes = total_mean_foxes /size

total_lower_quartile_times = np.zeros(1)
comm.Reduce(lower_quartile_times[-1] , total_lower_quartile_times , op=MPI.SUM , root=0)
total_lower_quartile_times = total_lower_quartile_times /size

total_upper_quartile_times = np.zeros(1)
comm.Reduce(upper_quartile_times[-1] , total_upper_quartile_times , op=MPI.SUM , root=0)
total_upper_quartile_times = total_upper_quartile_times /size

total_lower_quartile_foxes = np.zeros(1)
comm.Reduce(lower_quartile_foxes[-1] , total_lower_quartile_foxes , op=MPI.SUM , root=0)
total_lower_quartile_foxes = total_lower_quartile_foxes /size

total_upper_quartile_foxes = np.zeros(1)
comm.Reduce(upper_quartile_foxes[-1] , total_upper_quartile_foxes , op=MPI.SUM , root=0)
total_upper_quartile_foxes = total_upper_quartile_foxes /size

total_runs = np.zeros(1)
runs_MPI = np.ones(1)* runs
comm.Reduce(runs_MPI , total_runs , op=MPI.SUM , root=0)

total_dead_foxes  = np.zeros(1)
dead_foxes_MPI = np.ones(1)* dead_foxes 
comm.Reduce(dead_foxes_MPI , total_dead_foxes , op=MPI.SUM , root=0)

total_dead_everything  = np.zeros(1)
dead_everything_MPI = np.ones(1)* dead_everything 
comm.Reduce(dead_everything_MPI , total_dead_everything , op=MPI.SUM , root=0)

if rank ==0:

    print("Number of total runs {}".format(total_runs))

    print("Everything died {} times out of {} or {:.1f}%".format(total_dead_everything, total_runs, 100*total_dead_everything[0]/total_runs[0]))

    print("Foxes died {} times out of {} or {:.1f}%".format(total_dead_foxes, total_runs, 100*total_dead_foxes[0]/total_runs[0]))

    print("Second peak (days) is {:.1f} with IQR [{:.1f}-{:.1f}] ".format(total_mean_times[0], total_lower_quartile_times[0], total_upper_quartile_times[0]))

    print("Second peak (foxes) is {:.1f} with IQR [{:.1f}-{:.1f}] ".format(total_mean_foxes[0], total_lower_quartile_foxes[0], total_upper_quartile_foxes[0]))


"""        
if rank !=0:
	comm.Send(mean_times[-1] , dest=0, tag=111)
	comm.Send(mean_foxes[-1] , dest=0, tag=222)
	comm.send(runs , dest=0, tag=333)       
	#comm.send(dead_everything , dest=0, tag=444)  
	comm.send(dead_foxes , dest=0, tag=555)    
	comm.Send(lower_quartile_times[-1] , dest=0, tag=666)
	comm.Send(upper_quartile_times[-1] , dest=0, tag=777)
	comm.Send(lower_quartile_foxes[-1] , dest=0, tag=888)
	comm.Send(upper_quartile_foxes[-1] , dest=0, tag=999)
else:
    
    total_mean_times = mean_times[-1]
    total_mean_foxes = mean_foxes[-1]
    total_runs = runs
    #total_dead_everything = dead_everything
    total_dead_foxes = dead_foxes
	total_lower_quartile_times = lower_quartile_times[-1]
	total_upper_quartile_times = upper_quartile_times[-1]
	total_lower_quartile_foxes = lower_quartile_foxes[-1]
	total_upper_quartile_foxes = upper_quartile_foxes[-1]
    
    for i in range(1, size):
        comm.Recv(mean_times, ANY_SOURCE , tag=111)
        total_mean_times += mean_times

        comm.Recv(mean_foxes, ANY_SOURCE , tag=222)
        total_mean_foxes += mean_foxes    
   
        comm.recv(runs, ANY_SOURCE , tag=333)
        total_runs += runs

        #comm.recv(dead_everything, ANY_SOURCE , tag=444)
        #total_dead_everything += dead_everything      
  
        comm.recv(dead_foxes, ANY_SOURCE , tag=555)
        total_dead_foxes += dead_foxes      

        comm.Recv(lower_quartile_times, ANY_SOURCE , tag=666)
        total_lower_quartile_times += lower_quartile_times

        comm.Recv(upper_quartile_times, ANY_SOURCE , tag=777)
        total_upper_quartile_times += upper_quartile_times  

        comm.Recv(lower_quartile_foxes, ANY_SOURCE , tag=888)
        total_lower_quartile_foxes += lower_quartile_foxes

        comm.Recv(upper_quartile_foxes , ANY_SOURCE , tag=999)
        total_upper_quartile_foxes  += upper_quartile_foxes       
        
        
    print("Number of total runs {}".format(total_runs))
#    print("Everything died {} times out of {} or {:.1f}%".format(dead_everything, runs, 100*dead_everything/runs))
    print("Foxes died {} times out of {} or {:.1f}%".format(total_dead_foxes, total_runs, 100*total_dead_foxes/total_runs))
    print("Second peak (days) is {:.1f} with IQR [{:.1f}-{:.1f}] ".format(total_mean_times, total_lower_quartile_times, total_upper_quartile_times))
    print("Second peak (foxes) is {:.1f} with IQR [{:.1f}-{:.1f}] ".format(total_mean_foxes, total_lower_quartile_foxes, upper_quartile_foxes))
"""
"""
    plt.semilogx(mean_times,'-r')
    plt.semilogx(upper_quartile_times,':r')
    plt.semilogx(lower_quartile_times,':r')
    plt.ylabel('Second peak time (days)')
    plt.xlim(10)
    plt.show()



    plt.semilogx(mean_foxes,'-k')
    plt.semilogx(upper_quartile_foxes,':k')
    plt.semilogx(lower_quartile_foxes,':k')
    plt.ylabel('Second peak foxes')
    plt.xlim(10)
    plt.show()
"""


