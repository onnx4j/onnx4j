package org.onnx4j.opsets;

public class OperatorSets {
	
	private OperatorSet[] opsets;
	
	public static OperatorSets wrap(OperatorSet[] opsets) {
		return new OperatorSets(opsets);
	}
	
	private OperatorSets(OperatorSet[] opsets) {
		this.opsets = opsets;
	}
	
	/**
	 * 查找此版本实现下的Operator
	 * 
	 * @param opType
	 * @return
	 */
	public Operator getOperator(String opType) {
		for (OperatorSet opset : this.opsets) {
			Operator op = opset.getOp(opType);
			if (op == null)
				continue;
			
			return op;
		}
		
		return null;
	}

}
