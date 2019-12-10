package org.onnx4j.opsets.v7;

import java.util.Map;

import org.onnx4j.opsets.Operator;
import org.onnx4j.opsets.v6.OperatorSetSpecV6;
import org.onnx4j.opsets.v7.ops.AveragePoolV7;
import org.onnx4j.opsets.v7.ops.BatchNormalizationV7;
import org.onnx4j.opsets.v7.ops.DropoutV7;

/**
 * Default ONNX Operator Set in version 7
 * 
 * @author HarryLee
 *
 */
public interface OperatorSetSpecV7<T_TENSOR> extends OperatorSetSpecV6<T_TENSOR> {

	// public abstract AcosV7<T_TENSOR> getAcosV7();

	public abstract BatchNormalizationV7<T_TENSOR> getBatchNormalizationV7();

	public abstract DropoutV7<T_TENSOR> getDropoutV7();

	public abstract AveragePoolV7<T_TENSOR> getAveragePoolV7();

	@Override
	public default Map<String, Operator> initializeOperators() {
		Map<String, Operator> operators = OperatorSetSpecV6.super.initializeOperators();
		// 20191026
		// operators.put(AcosV7.OP_TYPE, this.getAcosV7());
		// 20191029
		operators.put(BatchNormalizationV7.OP_TYPE, this.getBatchNormalizationV7());
		// 20191120
		operators.put(DropoutV7.OP_TYPE, this.getDropoutV7());
		operators.put(AveragePoolV7.OP_TYPE, this.getAveragePoolV7());
		return operators;
	}

}
