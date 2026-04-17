package Algorithm.util.MtcscUtil;

import java.util.Comparator;

public class ComparatorTimestamp implements Comparator<TimePoint> {
	
	@Override
	public int compare(TimePoint tp1, TimePoint tp2) {
		// TODO Auto-generated method stub
        
		if (tp1.getTimestamp() > tp2.getTimestamp()) {    
            return 1;    
        } else if (tp1.getTimestamp() < tp2.getTimestamp()) {    
            return -1;    
        } else {    
            return 0;    
        }
	}
}
