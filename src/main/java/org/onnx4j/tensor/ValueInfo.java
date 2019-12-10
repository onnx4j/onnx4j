package org.onnx4j.tensor;

import org.onnx4j.onnx.prototypes.OnnxProto3.TensorProto;
import org.onnx4j.onnx.prototypes.OnnxProto3.TensorShapeProto;
import org.onnx4j.onnx.prototypes.OnnxProto3.ValueInfoProto;

public class ValueInfo {

	private DataType dataType;
	private Shape shape;

	public static ValueInfo toValueInfo(ValueInfoProto valueInfoProto) {
		TensorProto.DataType dataTypeProto = TensorProto.DataType
				.forNumber(valueInfoProto.getType().getTensorType()
						.getElemType());
		TensorShapeProto shapeProto = valueInfoProto.getType().getTensorType()
				.getShape();
		return new ValueInfo(DataType.from(dataTypeProto),
				Shape.toShape(shapeProto));
	}

	public ValueInfo(DataType dataType, Shape shape) {
		super();
		this.dataType = dataType;
		this.shape = shape;
	}

	public DataType getDataType() {
		return dataType;
	}

	public Shape getShape() {
		return shape;
	}
	
	public int getRank() {
		return this.shape.dims();
	}
	
	@Override
	public String toString() {
		return this.dataType + " -> " + this.shape;
	}

	@Override
	public boolean equals(Object o) {
		if (o != null && o instanceof ValueInfo) {
			return ((ValueInfo) o).getDataType().equals(this.dataType)
					&& ((ValueInfo) o).getShape().equals(this.shape);
		}
		return false;
	}

}
