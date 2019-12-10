package org.onnx4j.opsets.v1.ops;

import java.util.List;

import org.onnx4j.Inputs;
import org.onnx4j.Inputs.Input;
import org.onnx4j.Outputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.model.graph.node.Attributes;
import org.onnx4j.model.graph.node.attributes.IntAttribute;
import org.onnx4j.model.graph.node.attributes.IntsAttribute;
import org.onnx4j.opsets.v1.OperatorV1;
import org.onnx4j.tensor.DataType;

/**
 * Mul-1
 * 
 * @author HarryLee
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#MatMul
 * @version This version of the operator has been available since version 1 of
 *          the default ONNX operator set.
 *
 */
public interface MulV1<T_TENSOR> extends OperatorV1 {

	public static final String OP_TYPE = "Mul";

	public static final String ATTR_AXIS = "axis";

	public static final String ATTR_BROADCAST = "broadcast";

	public static final String ATTR_CONSUMED_INPUTS = "consumed_inputs";

	/**
	 * Performs element-wise binary multiplication (with limited broadcast
	 * support).
	 * 
	 * If necessary the right-hand-side argument will be broadcasted to match
	 * the shape of left-hand-side argument. When broadcasting is specified, the
	 * second tensor can either be of element size 1 (including a scalar tensor
	 * and any tensor with rank equal to or smaller than the first tensor), or
	 * having its shape as a contiguous subset of the first tensor's shape. The
	 * starting of the mutually equal shape is specified by the argument "axis",
	 * and if it is not set, suffix matching is assumed. 1-dim expansion doesn't
	 * work yet.
	 * 
	 * @constraints tensor(float16), tensor(float), tensor(double)
	 * @param x0
	 *            First operand, should share the type with the second operand.
	 * @param x1
	 *            Second operand. With broadcasting can be of smaller size than
	 *            A. If broadcasting is disabled it should be of the same size.
	 * @return Result, has same dimensions and type as A
	 */
	public abstract T_TENSOR mul(T_TENSOR a, T_TENSOR b, Long axis, Long broadcast, List<Long> consumedInputs);

	@Override
	public default DataType[] getTypeConstraints() {
		return DataType.floatTypes();
	}

	@Override
	public default OperatorStatus getStatus() {
		return OperatorStatus.STABLE;
	}

	@Override
	public default String getOpType() {
		return OP_TYPE;
	}

	@Override
	public default Outputs forward(Node node, Inputs inputs) {
		Attributes attrs = node.getAttrs();

		//
		// If set, defines the broadcast dimensions. See doc for details.
		//
		Long axis = attrs.getAttrValue(ATTR_AXIS, IntAttribute.class, null);

		//
		// Pass 1 to enable broadcasting (default is 0)
		//
		Long broadcast = attrs.getAttrValue(ATTR_AXIS, IntAttribute.class, 0L);

		//
		// Legacy optimization attribute.
		//
		List<Long> consumedInputs = attrs.getAttrValue(ATTR_CONSUMED_INPUTS, IntsAttribute.class, null);

		Input[] inputArray = inputs.get();
		return Outputs.wrap(node,
				this.mul(inputArray[0].getTensor(), inputArray[1].getTensor(), axis, broadcast, consumedInputs));
	}

}
