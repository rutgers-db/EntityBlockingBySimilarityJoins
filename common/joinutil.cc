/*
 * author: Dong Deng 
 * modified: Zhencan Peng in rutgers-db/RedPajama_Analysis
 * modified: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#include "common/joinutil.h"


/*
 * Setjoin
 */
void SetJoinUtil::printMemory()
{
#ifdef __linux__
	struct sysinfo memInfo;

	sysinfo(&memInfo);
	// long long totalVirtualMem = memInfo.totalram;
	// // Add other values in next statement to avoid int overflow on right hand
	// // side...
	// totalVirtualMem += memInfo.totalswap;
	// totalVirtualMem *= memInfo.mem_unit;

	// long long virtualMemUsed = memInfo.totalram - memInfo.freeram;
	// // Add other values in next statement to avoid int overflow on right hand
	// // side...
	// virtualMemUsed += memInfo.totalswap - memInfo.freeswap;
	// virtualMemUsed *= memInfo.mem_unit;
	// cout << "Total Virtual Memory: " << totalVirtualMem << endl;
	// cout << "Used Virtual Memory: " << virtualMemUsed << endl;

	long long totalPhysMem = memInfo.totalram;
	// Multiply in next statement to avoid int overflow on right hand side...
	totalPhysMem *= memInfo.mem_unit;

	long long physMemUsed = memInfo.totalram - memInfo.freeram;
	// Multiply in next statement to avoid int overflow on right hand side...
	physMemUsed *= memInfo.mem_unit;

	// cout << "Total Physical Memory: " << totalPhysMem << endl;
	std::cout << "Used Physical Memory: " << physMemUsed << std::endl;
#elif __APPLE__
	vm_size_t page_size;
	mach_port_t mach_port;
	mach_msg_type_number_t count;
	vm_statistics64_data_t vm_stats;

	mach_port = mach_host_self();
	count = sizeof(vm_stats) / sizeof(natural_t);
	if (KERN_SUCCESS == host_page_size(mach_port, &page_size) &&
		KERN_SUCCESS == host_statistics64(mach_port, HOST_VM_INFO,
											(host_info64_t)&vm_stats, &count)) {
		long long free_memory = (int64_t)vm_stats.free_count * (int64_t)page_size;

		long long used_memory =
			((int64_t)vm_stats.active_count + (int64_t)vm_stats.inactive_count +
			(int64_t)vm_stats.wire_count) *
			(int64_t)page_size;
		printf("free memory: %lld\nused memory: %lld\n", free_memory, used_memory);
	}
#endif
}


void SetJoinUtil::processMemUsage(double &vm_usage, double &resident_set)
{
	using std::ios_base;
	using std::ifstream;
	using std::string;

	vm_usage     = 0.0;
	resident_set = 0.0;

	// 'file' stat seems to give the most reliable results
	//
	ifstream stat_stream("/proc/self/stat",ios_base::in);

	// dummy vars for leading entries in stat that we don't care about
	//
	string pid, comm, state, ppid, pgrp, session, tty_nr;
	string tpgid, flags, minflt, cminflt, majflt, cmajflt;
	string utime, stime, cutime, cstime, priority, nice;
	string O, itrealvalue, starttime;

	// the two fields we want
	//
	unsigned long vsize;
	long rss;

	stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
				>> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
				>> utime >> stime >> cutime >> cstime >> priority >> nice
				>> O >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest

	stat_stream.close();

	long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
	vm_usage     = vsize / 1024.0;
	resident_set = rss * page_size_kb;
}


std::chrono::_V2::system_clock::time_point 
SetJoinUtil::logTime() 
{
	return std::chrono::high_resolution_clock::now();
}


double SetJoinUtil::repTime(const std::chrono::_V2::system_clock::time_point &start)
{
	auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    return duration.count() / 1000000.0;
}


int SetJoinParallelUtil::getHowManyThreads()
{
	int thread_num = 0;

    #pragma omp parallel for
    for(int i = 0; i<200;i++){
        #pragma omp critical
            thread_num = std::max(omp_get_thread_num(), thread_num);
    }

   return thread_num;
}


void SetJoinParallelUtil::printHowManyThreads()
{
	int thread_num = 0;

    #pragma omp parallel for
    for(int i = 0; i<200;i++){
        #pragma omp critical
            thread_num = std::max(omp_get_thread_num(),thread_num);
    }

    std::cout<<"max thread amount: "<<thread_num<<std::endl;
}


std::vector<int> 
SetJoinParallelUtil::getUniqueInts(const std::vector<std::pair<int, int>>& pairs)
{
	std::set<int> uniqueInts;
    for (const auto& p : pairs) {
        uniqueInts.insert(p.first);
        uniqueInts.insert(p.second);
    }

    // Convert set to vector
    std::vector<int> result(uniqueInts.begin(), uniqueInts.end());

    return result;
}


void SetJoinParallelUtil::mergeArrays(std::vector<std::vector<std::pair<int,int>>>* input, 
									  int arr_len, 
									  std::vector<std::vector<std::pair<int,int>>> & result)
{
	for(int i = 0; i < arr_len; i++){
        result.insert(result.end(), input[i].begin(), input[i].end());
    }
}


ui SetJoinParallelUtil::hval(const std::pair<ui, ui> &hf, ui &word)
{
	return hf.first * word + hf.second;
}


void SetJoinParallelUtil::generateHashFunc(ui seed, std::pair<ui, ui> &hf)
{
	srand(seed);
    unsigned int a = 0;
    while (a == 0)
        a = rand();
    unsigned int b = rand();
    hf.first = a;
    hf.second = b;
}


double SetJoinParallelUtil::shrinkBottomk(std::vector<std::vector<ui>>&  bottom_ks, double ratio)
{
	double average_size = 0;
    for(auto & vec:bottom_ks){
        unsigned int size = ceil(vec.size()*ratio);
        average_size += size;
        vec.resize(size);
    }
    average_size/= bottom_ks.size();
    return average_size;
}


template<typename T> bool
SetJoinParallelUtil::bottomKJaccard(const std::vector<T>& A, const std::vector<T>& B, double& thres)
{
	// Adpative K
    unsigned int k = min(A.size(),B.size());

    unsigned int posx =0;
    unsigned int posy = 0;
    
    unsigned int current_overlap = 0;
    unsigned int required_overlap = ceil(thres * k);
    unsigned int missing_limit = k - required_overlap;
    unsigned int missing_ele = 0;
    while (posx < k && posy < k) {

        // Check if the missing elements is more than the limit
        // Check if remaining elements are sufficient for required overlap
        if (missing_ele > missing_limit) return false;
        if (current_overlap >= required_overlap) return true;

        if (A[posx] == B[posy]) { 
            current_overlap++;
            posx++;
            posy++;
        } else if (A[posx] < B[posy]) {
            posx++;
            missing_ele++;
        } else {
            posy++;
            missing_ele++;
        }
    }
    return current_overlap >= required_overlap;
}


/*
 * String join
 */
bool StringJoinUtil::strLessT(const std::string &s1, const std::string &s2)
{
	if (s1.length() < s2.length()) 
		return true;
	else if (s1.length() > s2.length()) 
		return false;
	return s1 < s2;
}

uint64_t StringJoinUtil::strHash(const std::string &str, int stPos, int len)
{
	uint64_t __h = 0;
	int i = 0;
	while (i < len) 
		__h = (__h * stringHashNumber + str[stPos + i++]);
	return __h;
}

int StringJoinUtil::max(int a, int b, int c) 
{
  return (a >= b && a >= c) ? a : (b >= c ? b : c);
}

char StringJoinUtil::min(char a, char b, char c) 
{
  return (a <= b && a <= c) ? a : (b <= c ? b : c);
}

unsigned StringJoinUtil::min(unsigned a, unsigned b, unsigned c) 
{
  return (a <= b && a <= c) ? a : (b <= c ? b : c);
}

bool StringJoinUtil::PIndexLess(const PIndex &p1, const PIndex &p2)
{
	if (p1.stPos + p1.partLen < p2.stPos + p2.partLen) 
		return true;
	else if (p1.stPos + p1.partLen > p2.stPos + p2.partLen) 
		return false;

	if (p1.stPos < p2.stPos) 
		return true;
	else if (p1.stPos > p2.stPos) 
		return false;

	return false;
}